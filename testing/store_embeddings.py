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

# Load stop words
nltk.download('stopwords', download_dir=os.getcwd())
nltk.data.path = [os.getcwd()]
stop_words = set(stopwords.words('english'))

model = SentenceTransformer('all-MiniLM-L6-v2')


def clean_text(text):
    cleaned_text = text.lower()

    cleaned_text = cleaned_text.replace("story title: ", " ")
    cleaned_text = cleaned_text.replace("story text: ", " ")
    cleaned_text = cleaned_text.replace("--", " ")

    cleaned_text = ' '.join(_ for _ in cleaned_text.split() if _ not in stop_words)
    return cleaned_text


def generate_embeddings(data):
    cleaned_texts = [clean_text(story['text']) for story in data]
    embeddings = model.encode(cleaned_texts, show_progress_bar=True)
    return cleaned_texts, embeddings


def store_embeddings(hacker_news_stories, cleaned_texts, embeddings):

    # &&& delete me.
    # dates = [story["dt"][:10] for story in hacker_news_stories]
    # timestamps = [story["time"] for story in hacker_news_stories]

    chromadb_client = chromadb.PersistentClient(path="chromadb")

    # Check if the collection already exists and drop it
    if "hacker_news" in [collection.name for collection in chromadb_client.list_collections()]:
        chromadb_client.delete_collection("hacker_news")
    collection = chromadb_client.create_collection("hacker_news")

    for i, (story, cleaned_text, embedding) in enumerate(zip(hacker_news_stories, cleaned_texts, embeddings)):
        collection.add(
            metadatas=[{
                "date": story["dt"][:10],
                "timestamp": story["time"],
            }],
            # documents=[story["text"]],  # use raw text
            documents=[cleaned_text],  # use cleaned text ## if this is commented out, cleaned_text is not used.
            embeddings=[embedding.tolist()],
            ids=[f"id_{i}"],
        )

    # Check some embeddings.
    embeddings_ls = [embedding.tolist() for embedding in embeddings]
    results = collection.query(query_embeddings=[embeddings_ls[0]], n_results=1000)
    print(json.dumps(results))


if __name__ == '__main__':

    memory = Memory()
    memory.log_memory(print, "before")
    dt_start = datetime.now()

    file_path = 'hacker_news_stories.json'
    with open(file_path, 'r') as file:
        hacker_news_stories = json.load(file)

    cleaned_texts, embeddings = generate_embeddings(hacker_news_stories)
    memory.log_memory(print, "after_generate")

    print(f"type embeddings: {type(embeddings)}")
    print(f"len embeddings: {len(embeddings)}")
    print(f"size embeddings: {embeddings.size}")
    print(f"embeddings: {embeddings}")
    print(f"len embeddings 0: {len(embeddings[0])}")
    print(f"len embeddings 1: {len(embeddings[1])}")
    print(f"len embeddings 2: {len(embeddings[2])}")

    store_embeddings(hacker_news_stories, cleaned_texts, embeddings)
    memory.log_memory(print, "after_store")

    print(f"Total runtime: {datetime.now() - dt_start}")
