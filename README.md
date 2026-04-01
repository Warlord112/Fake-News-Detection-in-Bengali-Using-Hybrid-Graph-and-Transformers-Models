# Bengali Fake News Detection using BanglaBERT & GAT

This repository contains the codebase for detecting fake news in Bengali text using a hybrid architecture of a Transformer model (BanglaBERT) and a Graph Attention Network (GAT).

## Architecture Overview
1. **Feature Extraction:** A fine-tuned BanglaBERT model extracts 768-dimensional contextual embeddings from Bengali text.
2. **Graph Classification:** The extracted embeddings are constructed into a K-Nearest Neighbors graph using cosine similarity, and passed into a Graph Attention Network (GAT) using Focal Loss to classify the articles.

## Repository Structure
* `notebooks/`: Contains the original Kaggle notebook with data exploration and the full end-to-end pipeline.
* `src/`: Contains the core Python scripts for the pipeline.

## Pre-trained Models
Due to GitHub file size limits, the fine-tuned BanglaBERT model weights and `.pt` embedding files are not hosted in this repository. 
You can download the fully trained model weights here: **[ https://drive.google.com/drive/folders/1V7KuVdJ13a37xEJ-DuH_WuQg34-Kv3u4?usp=drive_link ]**

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train BanglaBERT & Extract Embeddings: `python src/01_train_banglabert.py`
3. Train the GAT model: `python src/02_train_gat.py`

### Option 2: Manual Execution via Jupyter Notebook
If you prefer to run the pipeline manually and view the step-by-step data exploration and visualizations, you can use the provided notebook:
* Open `notebooks/fake-news-detection-using-transformer (1).ipynb` in Jupyter Notebook, Google Colab, or Kaggle and execute the cells sequentially.