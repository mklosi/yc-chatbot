import chromadb
from datetime import datetime
import json


def query_by_date(collection, target_date):
    # Querying based on metadata (date-time field stored in metadatas)
    results = collection.query(
        where={"dt": {"$regex": f"^{target_date}"}},  # Use regex to match the date
        n_results=100  # Number of results to retrieve
    )
    return results


if __name__ == "__main__":
    # Initialize ChromaDB client pointing to the existing database directory
    chromadb_client = chromadb.PersistentClient(path="chromadb")

    # Load the existing collection by name
    collection = chromadb_client.get_collection("hacker_news")




    # Example usage to query for a specific date
    target_date = datetime.now().strftime("%Y-%m-%d")  # e.g., '2024-09-25'
    results = query_by_date(collection, target_date)

    # Print the results
    print(json.dumps(results, indent=2))
