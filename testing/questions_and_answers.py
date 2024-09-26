import chromadb
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import LongT5ForConditionalGeneration, T5Tokenizer

from testing.utils import model_name


def retrieve_relevant_documents(query, collection):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    n_results = 5

    query_embedding = model.encode([query], show_progress_bar=True)[0]
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    documents = [item for sublist in results["documents"] for item in sublist]
    return documents


# not used.
def generate_naive_answer(query, documents):

    model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    combined_input = query + "\n\n" + "\n\n".join(documents)
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=False)

    min_, max_ = 10, 100
    answer_ids = model.generate(inputs.input_ids, min_length=min_, max_length=max_)
    assert len(answer_ids) == 1
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    print(f"answer len with min_ '{min_}', max_ '{max_}': {len(answer)}")

    return answer


def generate_score_based_answer(query, documents):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    answers = []
    for doc in documents:
        # noinspection PyArgumentList
        answer = qa_pipeline(question=query, context=doc)
        answers.append(answer)

    best_answer = max(answers, key=lambda x: x['score'])
    return best_answer['answer']


if __name__ == "__main__":

    query = "Are there any news or events mentioning cyber security?"

    chromadb_client = chromadb.PersistentClient(path="chromadb")
    collection = chromadb_client.get_collection("hacker_news")

    relevant_documents = retrieve_relevant_documents(query, collection)

    # answer = generate_naive_answer(query, relevant_documents)
    answer = generate_score_based_answer(query, relevant_documents)

    print(f"Answer:\n{answer}")
