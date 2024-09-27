import json
import os
import warnings
from datetime import datetime

import chromadb
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from modules.memory import Memory

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load stop words. this needs to be outside functions so it doesn't download each time. duh!
nltk.download('stopwords', download_dir=os.getcwd())
nltk.data.path = [os.getcwd()]
stop_words = set(stopwords.words('english'))


def clean_text(text):

    cleaned_text = text.lower()

    cleaned_text = cleaned_text.replace("story title: ", " ")
    cleaned_text = cleaned_text.replace("story text: ", " ")
    cleaned_text = cleaned_text.replace("--", " ")

    cleaned_text = ' '.join(_ for _ in cleaned_text.split() if _ not in stop_words)
    return cleaned_text


def generate_embeddings(data):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    cleaned_texts = [clean_text(story['text']) for story in data]
    embeddings = model.encode(cleaned_texts, show_progress_bar=True)
    return cleaned_texts, embeddings


def store_embeddings(hacker_news_stories, use_cleaned_text, cleaned_texts, embeddings):

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
            documents=[cleaned_text if use_cleaned_text else story['text']],
            embeddings=[embedding.tolist()],
            ids=[f"id_{i}"],
        )

    # Check some embeddings.
    embeddings_ls = [embedding.tolist() for embedding in embeddings]
    # n_results=10 is only for sanity-checking the saved stories, so it doesn't matter.
    results = collection.query(query_embeddings=[embeddings_ls[0]], n_results=10)
    print(json.dumps(results))


if __name__ == '__main__':
    ## args being

    # Whether to use the cleaned text (remove stop words, lowecase, etc.), or use original text to store in chromadb.
    use_cleaned_text = True

    ## args end

    memory = Memory()
    memory.log_memory(print, "before")
    dt_start = datetime.now()

    file_path = 'hacker_news_stories.json'
    with open(file_path, 'r') as file:
        hacker_news_stories = json.load(file)
        print(f"Total stories to be stored: {len(hacker_news_stories)}")

    cleaned_texts, embeddings = generate_embeddings(hacker_news_stories)
    memory.log_memory(print, "after_generate")

    print(f"type embeddings: {type(embeddings)}")
    print(f"len embeddings: {len(embeddings)}")
    print(f"size embeddings: {embeddings.size}")
    print(f"embeddings: {embeddings}")
    print(f"len embeddings 0: {len(embeddings[0])}")
    print(f"len embeddings 1: {len(embeddings[1])}")
    print(f"len embeddings 2: {len(embeddings[2])}")

    store_embeddings(
        hacker_news_stories,
        use_cleaned_text,
        cleaned_texts,
        embeddings,
    )
    memory.log_memory(print, "after_store")

    print(f"Total runtime: {datetime.now() - dt_start}")
