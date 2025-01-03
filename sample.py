from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
from src.logger import setup_logger

import os

load_dotenv()
logger = setup_logger(__name__, int(os.getenv("LOG_LEVEL")))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

qwen_model = os.getenv("QWEN_MODEL")
qwen_api_key = os.getenv("QWEN_API_KEY")

def load_embedding_model():
    embedding_model = SentenceTransformer(os.path.abspath('./bge-small-zh-v1.5'))
    logger.info(f"Loaded bge-small-zh-v1.5 model, max input length: {embedding_model.max_seq_length}") 
    return embedding_model


def indexing_process(pdf_file, embedding_model):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=128
    )
    pdf_content_list = pdf_loader.load()
    pdf_text = "\n".join([page.page_content for page in pdf_content_list])
    logger.info(f"Total characters in PDF document: {len(pdf_text)}") 


    chunks = text_splitter.split_text(pdf_text)
    logger.info(f"Number of text chunks: {len(chunks)}") 

    # 文本块转化为嵌入向量列表，normalize_embeddings表示对嵌入向量进行归一化，用于准确计算相似度
    # 归一化是计算向量相似度时，将向量长度归一化到1，使得相似度计算更加准确。
    # 为什么需要归一化？
    # 因为向量长度不同，相似度计算结果会受到影响，归一化后，向量长度相同，相似度计算结果更准确。
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)

    logger.info("Text chunks converted to embedded vectors")

    # convert embedded vector lists into numpy arrays, FAISS indexing operations require numpy array inputs
    embeddings_np = np.array(embeddings)

    # get the dimension of the embedded vector (the length of each vector)
    dimension = embeddings_np.shape[1]

    # 使用余弦相似度创建FAISS索引
    # IndexFlatIP: 使用点积相似度（IP）作为相似度度量，计算两个向量之间的余弦相似度
    # IndexFlatL2: 使用欧几里得距离作为相似度度量，计算两个向量之间的欧几里得距离
    # IndexHNSW: 使用HNSW（Hierarchical Navigable Small World）算法，构建高效的近似最近邻搜索
    # IndexIVFFlat: 使用倒排文件（IVF）和点积相似度（IP），构建高效的近似最近邻搜索
    # IndexIVFPQ: 使用倒排文件（IVF）和乘积量化（PQ），构建高效的近似最近邻搜索
    # IndexPQ: 使用乘积量化（PQ），构建高效的近似最近邻搜索
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    logger.info("Indexing process completed")

    return index, chunks

def retrieval_process(query, index, chunks, embedding_model, top_k=3):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding])

    # 在 Faiss 索引中使用 query_embedding 进行搜索，检索出最相似的前 top_k 个结果。
    # 返回查询向量与每个返回结果之间的相似度得分（在使用余弦相似度时，值越大越相似）排名列表distances，最相似的 top_k 个文本块在原始 chunks 列表中的索引indices
    distances, indices = index.search(query_embedding, top_k)

    logger.info(f"Query: {query}")
    logger.info(f"Most similar top {top_k} text chunks:")

    results = []
    for i in range(top_k):
        result_chunk = chunks[indices[0][i]]
        logger.debug(f"Text chunk {i}:\n{result_chunk}") 

        result_distance = distances[0][i]
        logger.debug(f"Similarity score: {result_distance}\n")

        results.append(result_chunk)

    logger.info("Retrieval process completed")
    return results

def generate_process(query, chunks):
    llm_model = qwen_model
    dashscope.api_key = qwen_api_key

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"Reference document {i+1}: \n{chunk}\n\n"

    prompt = f"Answer the question based on the reference documents: {query}\n\n{context}"
    logger.debug(f"Prompt for the large model: \n\n{prompt}")

    messages = [{'role': 'user', 'content': prompt}]

    try:
        responses = dashscope.Generation.call(
            model = llm_model,
            messages=messages,
            result_format='message',
            stream=True,
            incremental_output=True
        )
        generated_response = ""
        logger.info("Generate process started")
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]['message']['content']
                generated_response += content
                print(content, end='')
            else:
                logger.error(f"Request failed: {response.status_code} - {response.message}")
                return None
        print("\n\n", end='')
        logger.info("Generate process completed")
        return generated_response
    except Exception as e:
        logger.error(f"Error occurred during large model generation: {e}")
        return None

def main():
    logger.info("RAG process started")

    query="请总结 2024 年 DevOps 的现状，以及未来发展趋势。"
    embedding_model = load_embedding_model()

    # indexing process: load pdf file, split text chunks, calculate embedded vectors, store in FAISS index (memory)
    index, chunks = indexing_process('./assets/2024-dora-accelerate-state-of-devops-report-zh-cn-martinliu.pdf', embedding_model)

    # retrieval process: convert user query to embedded vector, retrieve most similar text chunks
    retrieval_chunks = retrieval_process(query, index, chunks, embedding_model)

    # generate process: call Qwen large model to generate response
    generate_process(query, retrieval_chunks)

    logger.info("RAG process completed")

if __name__ == "__main__":
    main()
