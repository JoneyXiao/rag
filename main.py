from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.text_splitter import (
#     CharacterTextSplitter,
#     RecursiveCharacterTextSplitter,
#     MarkdownTextSplitter,
#     PythonCodeTextSplitter,
#     LatexTextSplitter,
#     SpacyTextSplitter,
#     NLTKTextSplitter
# )
from sentence_transformers import SentenceTransformer
import os
import dashscope
from http import HTTPStatus
import chromadb
import uuid
import shutil
from dotenv import load_dotenv
from src.logger import setup_logger
from rank_bm25 import BM25Okapi  # alternative: langchain.retrievers.EnsembleRetriever
import jieba
from FlagEmbedding import FlagReranker

load_dotenv()
# Set up logger
logger = setup_logger(__name__, int(os.getenv("LOG_LEVEL")))
jieba.setLogLevel(20)  # INFO

# Dont use word parallelization to avoid conflicts or deadlocks caused by
# running multiple models in a multi-threaded or multi-process environment.
# bge-small-zh-v1.5 pytorch model
os.environ["TOKENIZERS_PARALLELISM"] = "false"

QWEN_MODEL = os.getenv("QWEN_MODEL")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

RETRIEVAL_TOP_K = 1
RERANKING_TOP_K = 3
SPLIT_CHUNK_SIZE = 1024
SPLIT_CHUNK_OVERLAP = 256
EMBEDDING_MODEL_PATH = "./bge-small-zh-v1.5"
# Select the corresponding document parser loader class and input parameters based on the document type
DOCUMENT_LOADER_MAPPING = {
    ".pdf": (PDFPlumberLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".xml": (UnstructuredXMLLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
}


def load_document(file_path: str) -> str:
    """
    Parses files in multiple document formats and returns document content strings
    :param file_path: document file path
    :return: document content strings
    """
    ext = os.path.splitext(file_path)[1]
    loader_class, loader_args = DOCUMENT_LOADER_MAPPING.get(ext, (None, None))

    if loader_class:
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        logger.info(f"Loaded document {file_path} with {len(content)} characters")
        return content

    logger.warning(f"Unsupported document type: '{ext}'")
    return ""


def load_embedding_model(model_path: str = EMBEDDING_MODEL_PATH) -> SentenceTransformer:
    """
    Loads a pre-trained sentence transformer model for embedding generation.
    :param model_path: Path to the pre-trained sentence transformer model
    :return: Loaded sentence transformer model
    """
    try:
        model_name = os.path.basename(model_path)
        embedding_model = SentenceTransformer(os.path.abspath(model_path))
        logger.info(
            f"Loaded model {model_name} with max sequence length(max input length): {embedding_model.max_seq_length}"
        )
        return embedding_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e


def reranking(query: str, chunks: list[str], top_k: int = RERANKING_TOP_K) -> list[str]:
    # initialize reranking model, use fp16 to accelerate, cache_dir='~/.cache/huggingface/hub'
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    # construct input pairs, each query and chunk form a pair
    input_pairs = [[query, chunk] for chunk in chunks]

    # compute semantic similarity score between each chunk and query
    scores = reranker.compute_score(input_pairs, normalize=True)

    logger.info(f"Document chunks reranking scores: {scores}")

    # sort scores and get top_k chunks
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranking_chunks = [chunks[i] for i in sorted_indices[:top_k]]

    # print top_k chunks
    for i in range(top_k):
        logger.info(
            f"Reranked document chunk {i+1}: Similarity score: {scores[sorted_indices[i]]}"
        )
        logger.debug(f"Document chunk info: \n\n{reranking_chunks[i]}\n")

    return reranking_chunks


def indexing_process(
    folder_path: str,
    embedding_model: SentenceTransformer,
    collection: chromadb.PersistentClient,
) -> None:
    """
    Indexes documents in a folder by splitting them into chunks and generating embeddings.
    :param folder_path: Path to the folder containing documents
    :param embedding_model: Pre-loaded sentence transformer model for embedding generation
    :param collection: ChromaDB collection to store embeddings and documents
    :return: None
    """
    all_chunks = []
    all_ids = []

    logger.info("[Indexing process] start ...")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            document_text = load_document(file_path)
            if document_text:
                logger.info(
                    f"Document {filename} character count: {len(document_text)}"
                )
                # text_splitter = SpacyTextSplitter(
                #    chunk_size=512, chunk_overlap=128, pipeline="zh_core_web_sm"
                # )

                # configure RecursiveCharacterTextSplitter to split text chunks, each chunk size is 512 characters (non-token),
                # and the overlap between adjacent chunks is 128 characters (non-token)
                # can replace with CharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter, LatexTextSplitter, NLTKTextSplitter, etc
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=SPLIT_CHUNK_SIZE, chunk_overlap=SPLIT_CHUNK_OVERLAP
                )
                chunks = text_splitter.split_text(document_text)
                logger.info(f"Document {filename} chunk count: {len(chunks)}")

                all_chunks.extend(chunks)
                all_ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])

    # convert chunks to embedding vectors, normalize_embeddings is used to normalize the embedding vectors for accurate similarity calculation
    # normalization is used to calculate the similarity between vectors by normalizing the vector length to 1, making the similarity calculation more accurate.
    # why need normalization?
    # because the vector length is different, the similarity calculation result will be affected,
    # after normalization, the vector length is the same, the similarity calculation result is more accurate.
    embeddings = [
        embedding_model.encode(chunk, normalize_embeddings=True).tolist()
        for chunk in all_chunks
    ]

    collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks)
    logger.info("Embeddings generated and stored in vector database")
    logger.info("[Indexing process] completed")


def retrieval_process(
    query: str,
    collection: chromadb.PersistentClient,
    embedding_model: SentenceTransformer = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    """
    Retrieves similar chunks from a vector database based on a query.
    :param query: The query string to search for
    :param collection: The ChromaDB collection to search in
    :param embedding_model: The embedding model to use for encoding the query
    :param top_k: The number of top results to return
    :return: A list of similar chunks
    """
    logger.info("[Retrieval process] start ...")
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding], n_results=top_k
    )

    # get all documents from Chroma collection
    all_docs = collection.get()["documents"]

    # tokenize all documents
    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]

    # initialize BM25Okapi using tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)

    # tokenize query
    tokenized_query = list(jieba.cut(query))

    # calculate the BM25 score of the query statement with respect to each document
    # return a relevance score for each document
    bm25_scores = bm25.get_scores(tokenized_query)

    # get top k indices of documents with highest BM25 scores
    bm25_top_k_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:top_k]
    # get corresponding documents
    bm25_chunks = [all_docs[i] for i in bm25_top_k_indices]

    # print vector search results
    logger.info(f"Query: {query}")
    logger.info(f"Top {top_k} most similar chunks by vector search:")
    vector_chunks = []
    for rank, (doc_id, doc) in enumerate(
        zip(vector_results["ids"][0], vector_results["documents"][0])
    ):
        logger.info(f"Vector search rank: {rank + 1}")
        logger.info(f"Chunk ID: {doc_id}")
        logger.debug(f"Chunk info:\n\n{doc}\n")
        vector_chunks.append(doc)

    # print BM25 search results
    logger.info(f"Top {top_k} most similar chunks by BM25 search:")
    for rank, doc in enumerate(bm25_chunks):
        logger.info(f"BM25 search rank: {rank + 1}")
        logger.debug(f"Chunk info:\n\n{doc}\n")

    # use reranking model to rerank the search results
    reranking_chunks = reranking(query, list(set(vector_chunks + bm25_chunks)), top_k)

    logger.info("[Retrieval process] completed")

    return reranking_chunks


def generate_process(query: str, chunks: list[str]) -> str:
    """
    Generates a response to a query using a large language model.
    :param query: The query string to search for
    :param chunks: The chunks to use for generating the response
    :return: The generated response
    """
    llm_model = QWEN_MODEL
    dashscope.api_key = QWEN_API_KEY

    logger.info("[Generation process] start ...")
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"参考文档{i+1}: \n{chunk}\n\n"

    prompt = f"根据参考文档回答问题：{query}\n\n{context}"
    logger.debug(f"Generation prompt:\n\n{prompt}")

    messages = [{"role": "user", "content": prompt}]

    try:
        responses = dashscope.Generation.call(
            model=llm_model,
            messages=messages,
            result_format="message",
            stream=True,
            incremental_output=True,
        )
        generated_response = ""
        last_response = None
        logger.info("Response from LLM:\n")
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]["message"]["content"]
                generated_response += content
                print(content, end="")  # Keep real-time output
                last_response = response.usage
            else:
                logger.error(
                    f"Request failed: {response.status_code} - {response.message}"
                )
                return None
        print("\n\n", end="")
        if last_response:
            logger.info(
                f"Input Tokens: {last_response.input_tokens}, Output Tokens: {last_response.output_tokens}, Total Tokens: {last_response.total_tokens}"
            )
        logger.info("[Generation process] completed")
        return generated_response
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return None


def main():
    logger.info("🚀 Starting RAG process ...")

    chroma_db_path = os.path.abspath("./chroma_db")
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)
        logger.info(f"Cleared existing ChromaDB directory")

    client = chromadb.PersistentClient(chroma_db_path)
    collection = client.get_or_create_collection(name="documents")
    logger.info("Created ChromaDB collection")

    embedding_model = load_embedding_model()

    indexing_process("./assets", embedding_model, collection)
    query = "请简要总结 2024 年 DevOps 领域的发展趋势。"
    retrieval_chunks = retrieval_process(query, collection, embedding_model)
    generate_process(query, retrieval_chunks)
    logger.info("✅ RAG process completed")


if __name__ == "__main__":
    main()
