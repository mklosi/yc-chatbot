import random
from collections import defaultdict
from datetime import datetime

import chromadb
from sympy import ceiling
from transformers import LongT5ForConditionalGeneration, T5Tokenizer

from modules.memory import Memory
from modules.utils import model_name

memory = Memory()


# not used.
def get_chunks(big_text, chunk_size, overlap_percent):
    words = big_text.split()  # Split the text into words
    chunks = []
    overlap_size = ceiling(chunk_size * overlap_percent)  # Calculate the overlap size
    step_size = chunk_size - overlap_size  # Calculate the step size between chunks
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunk_text = ' '.join(chunk)
        chunks.append(chunk_text)
        start += step_size
        if start >= len(words):
            break

    return chunks


def summarize_text(
    text, tokenizer, max_input_tokens, truncation, model, min_output_tokens, max_output_tokens,
):

    # memory.log_memory(print, "tokenizer_before")
    # tokenizer_start_dt = datetime.now()
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_tokens, truncation=truncation)
    # memory.log_memory(print, "tokenizer_after")
    # print(f"tokenizer_runtime: {datetime.now() - tokenizer_start_dt}")

    # print(f"Generated '{inputs['input_ids'].size(1)} tokens.'")

    # memory.log_memory(print, "generate_before")
    # generate_start_dt = datetime.now()
    summary_ids = model.generate(inputs.input_ids, min_length=min_output_tokens, max_length=max_output_tokens)
    # memory.log_memory(print, "generate_after")
    # print(f"generate_runtime: {datetime.now() - generate_start_dt}")

    assert len(summary_ids) == 1
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # print(f"summary len with min_ '{min_output_tokens}', max_ '{max_output_tokens}': {len(summary)}\n")

    return summary, inputs['input_ids'].size(1)


if __name__ == "__main__":
    ## args begin

    # All the arguments bellow are simply parameters to fine tune the
    #   generation of the summaries, so they are reasonable, and it can finish on my
    #   local laptop in a reasonable amount of time.

    max_docs_per_day = 25

    doc_max_input_tokens = 10000
    doc_truncation = True
    doc_min_output_tokens = 10
    doc_max_output_tokens = 1000

    tot_max_input_tokens = 25000
    tot_truncation = True
    tot_min_output_tokens = 15
    tot_max_output_tokens = 1500

    ## args end

    start_dt = datetime.now()

    chromadb_client = chromadb.PersistentClient(path="chromadb")
    collection = chromadb_client.get_collection("hacker_news")
    # results = collection.get(where={"date": {"$eq": query_date_arg_str}})
    results = collection.get()

    # # Print the results
    # print(json.dumps(results))

    date_to_documents = defaultdict(list)

    for metadata, doc in zip(results["metadatas"], results["documents"]):
        date = metadata["date"]
        date_to_documents[date].append(doc)

    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    # Generate reports for each day by first summarizing the individual documents, then summarizing the summaries.
    for date, doc_ls in sorted(date_to_documents.items()):

        date_start_dt = datetime.now()

        print(f"\n--- Start report for date: {date} ---------------------\n")

        # not used.
        # chunks = get_chunks(big_text, chunk_size=max_length, overlap_percent=0.33)

        random.shuffle(doc_ls)  # shuffle the documents, since we are taking only a subset.
        max_docs_per_day = min(max_docs_per_day, len(doc_ls))
        doc_ls = doc_ls[:max_docs_per_day]

        summaries = []
        for idx, doc in enumerate(doc_ls):
            doc_start_dt = datetime.now()
            individual_summary, token_count = summarize_text(
                doc,
                tokenizer,
                doc_max_input_tokens,
                doc_truncation,
                model,
                doc_min_output_tokens,
                doc_max_output_tokens,
            )
            summaries.append(individual_summary)
            print(f"Processed document '{idx}' with '{token_count}' token_count in: {datetime.now() - doc_start_dt}")

        big_text = " ".join(summaries)
        summary, token_count = summarize_text(
            big_text,
            tokenizer,
            tot_max_input_tokens,
            tot_truncation,
            model,
            tot_min_output_tokens,
            tot_max_output_tokens,
        )
        print(f"\nProcessed date '{date}' with '{token_count}' token_count in: {datetime.now() - date_start_dt}")

        print(f"\nSummary Report for '{date}':\n\n{summary}")

    print(f"\nTotal runtime: {datetime.now() - start_dt}")
