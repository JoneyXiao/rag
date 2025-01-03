# Setup

```bash
cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

sudo apt-get install libreoffice # Linux
brew install --cask libreoffice # MacOS
```

# Formatting

(Black formatter)[https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter]

# Run

```bash
source venv/bin/activate
python main.py
```

# Development

```
# For Mac
brew install git-lfs

# For Ubuntu/Debian
sudo apt-get install git-lfs

# For Windows (using Chocolatey)
choco install git-lfs

# Setup Git LFS in your repository
git lfs install

# Track your large files (add patterns to .gitattributes)
git lfs track "*.pdf"
git lfs track "*.model"
git lfs track "*.bin"

git add .gitattributes

git commit -m "add file.pdf"
git push
```