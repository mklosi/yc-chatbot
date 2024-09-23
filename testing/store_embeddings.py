import json
import os
import sys
import warnings
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from testing.memory import Memory
import chromadb
from chromadb.config import Settings

warnings.simplefilter(action='ignore', category=FutureWarning)

# # Load stop words
# nltk.download('stopwords', download_dir=os.getcwd())
# nltk.data.path = [os.getcwd()]
# stop_words = set(stopwords.words('english'))

model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text(text):
    # Convert to lowercase
    clean_text = text.lower()

    # # Remove stop words # &&&
    # clean_text = ' '.join(_ for _ in clean_text.split() if _ not in stop_words)

    return clean_text


def generate_embeddings(data):
    texts = [preprocess_text(story['text']) for story in data]
    embeddings = model.encode(texts, show_progress_bar=True)
    return texts, embeddings


def store_embeddings(dts, timestamps, texts, embeddings):
    chromadb_client = chromadb.PersistentClient(path="chromadb")

    # Check if the collection already exists and drop it
    if "hacker_news" in [collection.name for collection in chromadb_client.list_collections()]:
        chromadb_client.delete_collection("hacker_news")
    collection = chromadb_client.create_collection("hacker_news")

    for i, (dt, timestamp, text, embedding) in enumerate(zip(dts, timestamps, texts, embeddings)):
        collection.add(
            metadatas=[{"dt": dt, "timestamp": timestamp}],
            documents=[text],
            embeddings=[embedding.tolist()],
            ids=[f"id_{i}"],
        )

    # Check some embeddings.
    embeddings_ls = [embedding.tolist() for embedding in embeddings]
    results = collection.query(query_embeddings=[embeddings_ls[0]], n_results=5)
    print(json.dumps(results))


if __name__ == '__main__':

    memory = Memory()
    memory.log_memory(print, "before")
    dt_start = datetime.now()

    file_path = 'hacker_news_stories.json'
    with open(file_path, 'r') as file:
        hacker_news_stories = json.load(file)

    texts, embeddings = generate_embeddings(hacker_news_stories)
    memory.log_memory(print, "after_generate")

    print(f"type embeddings: {type(embeddings)}")
    print(f"len embeddings: {len(embeddings)}")
    print(f"size embeddings: {embeddings.size}")
    print(f"embeddings: {embeddings}")
    print(f"len embeddings 0: {len(embeddings[0])}")
    print(f"len embeddings 1: {len(embeddings[1])}")
    print(f"len embeddings 2: {len(embeddings[2])}")

    dts = [story["dt"] for story in hacker_news_stories]
    timestamps = [story["time"] for story in hacker_news_stories]
    store_embeddings(dts, timestamps, texts, embeddings)
    memory.log_memory(print, "after_store")

    print(f"Total runtime: {datetime.now() - dt_start}")
