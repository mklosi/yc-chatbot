import os
import time
from datetime import datetime, timedelta
import datetime as dtt
import pytz
import json
from requests.exceptions import HTTPError, SSLError

import requests
from bs4 import BeautifulSoup

# tz_str = "UTC"
tz_str = "US/Pacific"
# tz_str = "America/Los_Angeles"
os.environ['TZ'] = tz_str
time.tzset()
tz_ = pytz.timezone(tz_str)


def extract_text_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an error for bad status codes
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from the main content of the page
    # This may vary depending on the structure of the webpage
    paragraphs = soup.find_all('p')
    page_text = ' '.join([p.get_text() for p in paragraphs])
    return page_text


def fetch_hacker_news_stories(
    store_every,
    stories_cutoff_in_days,
    min_chars_per_story,
):
    start_dt = datetime.now()
    base_url = "https://hacker-news.firebaseio.com/v0"
    max_item_id = requests.get(f"{base_url}/maxitem.json", timeout=10).json()

    # Get the current time and calculate the cutoff time for the 'days' parameter
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(days=stories_cutoff_in_days)
    # The cutoff day should be full, and included.
    cutoff_time = cutoff_time.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_timestamp = int(cutoff_time.timestamp())

    stories = []
    for item_id in range(max_item_id, 0, -store_every):

        item_url = f"{base_url}/item/{item_id}.json"
        try:
            item_data = requests.get(item_url, timeout=10).json()
        except requests.exceptions.Timeout:
            print(f"WARNING: Timeout for item_data for item_id: {item_id}")
            continue
        except requests.exceptions.ConnectionError:
            print(f"WARNING: ConnectionError for item_data for item_id: {item_id}")
            continue

        # Check if the item's time is less then the cutoff. If that's the case,
        #   we can break early, since we know that any stories fetched from now
        #   on will have a 'time' less than the cutoff as well, since we started
        #   with the item with the largest item_id and going backwards.
        if item_data['time'] < cutoff_timestamp:
            print("Reached cutoff time.")
            break

        # Skip this item if it's not a story or if it doesn't have an URL.
        if item_data['type'] != 'story' or 'url' not in item_data:
            continue

        # Try fetching and extract text from the URL. If HTTPError is raised, skip.
        try:
            item_data['text'] = (
                f"Story Title: {item_data['title']} -- Story text: {extract_text_from_url(item_data['url'])}"
            )
        except HTTPError:
            print(f"WARNING: HTTPError for story_id: {item_id}")
            continue
        except SSLError:
            print(f"WARNING: SSLError for story_id: {item_id}")
            continue
        except requests.exceptions.Timeout:
            print(f"WARNING: Timeout for extract_text for story_id: {item_id}")
            continue
        except requests.exceptions.ConnectionError:
            print(f"WARNING: ConnectionError for extract_text for item_id: {item_id}")
            continue

        # Include only stories that have at least some characters.
        if len(item_data['text']) < min_chars_per_story:
            print(f"WARNING: Min char reqs not met for story_id: {item_id}")
            continue

        # Add datetime field based on timestamp.
        item_data["dt"] = datetime.fromtimestamp(item_data["time"], tz=tz_).isoformat()

        stories.append(item_data)

        # Checkpointer.
        if len(stories) % 10 == 0:
            print(f"Stories fetched so far: {len(stories)}")

    print(f"Total stories fetched: {len(stories)}")
    print(f"Runtime: {datetime.now() - start_dt}")

    return stories


if __name__ == '__main__':

    store_every = 1
    stories_cutoff_in_days = 3
    min_chars_per_story = 100

    hacker_news_stories = fetch_hacker_news_stories(
        store_every=store_every,
        stories_cutoff_in_days=stories_cutoff_in_days,
        min_chars_per_story=min_chars_per_story,
    )

    file_path = 'hacker_news_stories.json'
    with open(file_path, 'w') as json_file:
        # noinspection PyTypeChecker
        json.dump(hacker_news_stories, json_file)
    print(f"Stories written to {file_path}")
