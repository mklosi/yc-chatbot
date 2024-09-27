import chromadb
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import LongT5ForConditionalGeneration, T5Tokenizer

from modules.utils import model_name


def retrieve_relevant_documents(query, collection, top_n_results):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode([query], show_progress_bar=True)[0]
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_n_results
    )
    documents = [item for sublist in results["documents"] for item in sublist]
    return documents


# not used.
def generate_naive_answer(query, documents, min_output_tokens, max_output_tokens):

    model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    combined_input = query + "\n\n" + "\n\n".join(documents)
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=False)

    answer_ids = model.generate(inputs.input_ids, min_length=min_output_tokens, max_length=max_output_tokens)
    assert len(answer_ids) == 1
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    print(f"answer len with min_ '{min_output_tokens}', max_ '{max_output_tokens}': {len(answer)}")

    return answer


def generate_score_based_answer(query, documents):

    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0)

    answers = []
    for doc in documents:
        # noinspection PyArgumentList
        answer = qa_pipeline(question=query, context=doc)
        answers.append(answer)

    best_answer = max(answers, key=lambda x: x['score'])
    return best_answer['answer']


if __name__ == "__main__":

    # query = "Are there any news or events mentioning cyber security?"
    query = "What are some news related to Joe Biden or Donald Trump recently?"

    top_n_results = 20

    chromadb_client = chromadb.PersistentClient(path="chromadb")
    collection = chromadb_client.get_collection("hacker_news")

    relevant_documents = retrieve_relevant_documents(
        query,
        collection,
        top_n_results,
    )

    # answer = generate_naive_answer(query, relevant_documents, min_output_tokens=10, max_output_tokens=100)
    answer = generate_score_based_answer(query, relevant_documents)

    print(f"Answer:\n{answer}")
