from collections import defaultdict

import chromadb
import json
from transformers import LongT5ForConditionalGeneration, T5Tokenizer

from testing.utils import model_name


def summarize_text(text):

    # max_length = 16384 # &&&
    # if len(text) > max_length:
    #     raise ValueError(f"Text length of '{len(text)}' longer than '{max_length}'.") &&&

    model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)

    min_, max_ = 10, 500
    summary_ids = model.generate(inputs.input_ids, min_length=min_, max_length=max_)
    assert len(summary_ids) == 1
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(f"summary len with min_ '{min_}', max_ '{max_}': {len(summary)}")

    return summary


def save_report(report, date):
    filename = f"summary_report_{date}.txt"
    with open(filename, "w") as file:
        file.write(report)
    print(f"Report saved to {filename}")


if __name__ == "__main__":

    chromadb_client = chromadb.PersistentClient(path="chromadb")
    collection = chromadb_client.get_collection("hacker_news")
    # results = collection.get(where={"date": {"$eq": query_date_arg_str}}) &&&
    results = collection.get()

    # Print the results
    print(json.dumps(results))

    date_to_documents = defaultdict(list)

    for metadata, doc in zip(results["metadatas"], results["documents"]):
        date = metadata["date"]
        date_to_documents[date].append(doc)

    # Generate reports for each day
    for date, doc_ls in sorted(date_to_documents.items()):

        # if date != "2024-09-23":
        #     print(f"Skipping date: {date}")  # &&&

        print(f"\n--- Summary Report for: {date} ---------------------\n")

        text = " -- ".join(doc_ls)
        summary = summarize_text(text)

        print(f"\n{summary}")
