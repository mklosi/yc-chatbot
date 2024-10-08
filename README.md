# yc-chatbot

A RAG application using open-source tools like LlamaIndex to ingest, summarize, and query Hacker News events stored in a vector database.

## Setup Instructions

Assuming you are running this application in an environment where the `brew` tool is installed. 

* First run `brew install direnv` - "direnv" is a tool that allows you to automatically set up some things whenever you `cd` into the project's directory. The things that are run are described in [.envrc](.envrc). Any shell command listed there will run when you `cd yc-chatbot`. The project root will be automatically added to the environment variable PYTHONPATH.

* You will also need [poetry](https://python-poetry.org/). "poetry" is a python environment management tool that makes it incredibly easy to resolve dependencies and install them. Poetry allows you to "lock" package versions at a given time and install the same versions consistently as long as the ".lock" file is untouched. The "[poetry.lock](poetry.lock)" file has already been created and checked in this repo for your convenience, and the `poetry install` command will run automatically through "direnv". You will need **python3.10** installed on your machine.

After you have set up the environment, you can run the following 4 modules in the defined order:

## 4 modules

### [fetch_stories.py](modules/fetch_stories.py)

Run with `python3 modules/fetch_stories.py`

This module is responsible for fetching Hacker News stories for the last N days. The stories are stored in JSON, later to be ingested into ChromaDB. The module should be fairly robust, as it handles many halting exceptions.

This tool, and the other 3, take parameters, but not using the traditional way using python's `argparse` built-in module. The parameters are just variables defined inside each script's `__main__` method. Using `argparse` under this testing context is a bit of a time-waster.

#### Parameters:
* **store_every** (int) - When fetching stories from HN for each day, store only once every _this_ number. If this number is 1, then store every story. I added this param, since processing every story for the last 3 days takes about 6h to complete. With this parameter we can skip a certain number of them for testing purposes.
* **stories_cutoff_in_days** (int) - The amount of days to go back in time and fetch stories.
* **min_chars_per_story** (int) - Don't save stories that have fewer than _this_ chars.

### [store_embeddings.py](modules/store_embeddings.py)

This module is responsible for reading all the JSON stories fetched by the previous module and store them, along with metadata, into a ChromaDB collection.

#### Parameters:
* **use_cleaned_text** (bool) - Whether to use the cleaned text (remove stop words, lowercase, etc.), or use original text to store in ChromaDB.

### [generate_reports.py](modules/generate_reports.py)

This module is responsible for generating summary reports for each day of the ingested Hacker News data. 

This module first retrieves documents from ChromaDB for each day and summarizes each document individually using a transformer model. After summarizing individual documents, it concatenates these summaries and generates a final summary report for each day. The summarization process can be fine-tuned using the parameters mentioned below to control input/output token lengths and document truncation.

#### Parameters:
* **max_docs_per_day** (int): The maximum number of documents to process per day.
* **doc_max_input_tokens** (int): Maximum number of tokens allowed for each document summary input.
* **doc_truncation** (bool): Whether to truncate documents that exceed the token limit.
* **doc_min_output_tokens** (int): Minimum number of tokens for the output summary of each document.
* **doc_max_output_tokens** (int): Maximum number of tokens for the output summary of each document.
* **tot_max_input_tokens** (int): Maximum number of tokens allowed for the combined summaries when generating the final daily report.
* **tot_truncation** (bool): Whether to truncate the combined summaries that exceed the token limit.
* **tot_min_output_tokens** (int): Minimum number of tokens for the final daily report summary.
* **tot_max_output_tokens** (int): Maximum number of tokens for the final daily report summary.

### [questions_and_answers.py](modules/questions_and_answers.py)

This module allows you to query the ingested Hacker News stories and retrieve answers based on the content. 

The module offers two methods for generating answers:
* **Naive Answer Generation**: Combines the query with the most relevant documents retrieved from ChromaDB and generates an answer using a transformer model. The answer length can be adjusted with minimum and maximum token limits.
* **Score-Based Answer Generation**: Utilizes a pre-trained question-answering pipeline to find the best answer from the relevant documents. It retrieves the top N most relevant documents, scores them based on their relevance to the query, and returns the highest-scoring answer.

#### Parameters:
* **query** (str): The question you want to ask based on the ingested news stories.
* **top_n_results** (int): The number of top relevant documents to retrieve from ChromaDB. 

## Further Improvements

* [fetch_stories.py](modules/fetch_stories.py) takes a long time to complete with very little mem footprint. Both the fetching of the HN items from the HN API, and the scraping of the html webpages associated with each item (story) can be parallelized using something like `aiohttp` and `asyncio`.
* In [generate_reports.py](modules/generate_reports.py), currently I summarize each individual article separately and then summarize all summaries for each day. Running the summary generation on all the stories for each day at once, would probably yield better results, but I can't do that on my local machine. Probably the next step would be to move the code in GCP or AWS and use G5 or G6 instances to run it. 
* In the [questions_and_answers.py](modules/questions_and_answers.py) module, I could explore more advanced models for the retrieval step, such as `all-mpnet-base-v2` or `all-distilroberta-v1`, which generally provide better semantic representations and, hence, better retrieval quality.
