import os
import time
from datetime import datetime, timedelta
import datetime as dtt
import pytz
import json
from requests.exceptions import HTTPError

import requests
from bs4 import BeautifulSoup

# tz_str = "UTC"
tz_str = "US/Pacific"
# tz_str = "America/Los_Angeles"
os.environ['TZ'] = tz_str
time.tzset()
tz_ = pytz.timezone(tz_str)


def extract_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from the main content of the page
    # This may vary depending on the structure of the webpage
    paragraphs = soup.find_all('p')
    page_text = ' '.join([p.get_text() for p in paragraphs])
    return page_text


def fetch_hacker_news_stories(
    max_stories,
    stories_cutoff_in_days,
    min_chars_per_story,
):
    start_dt = datetime.now()
    base_url = "https://hacker-news.firebaseio.com/v0"
    top_stories_url = f"{base_url}/newstories.json"

    # Get the current time and calculate the cutoff time for the 'days' parameter
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(days=stories_cutoff_in_days)
    cutoff_timestamp = int(cutoff_time.timestamp())

    # Fetch the top stories. Gets up to 500 stories.
    story_ids = requests.get(top_stories_url).json()

    stories = []
    for story_id in story_ids:
        story_url = f"{base_url}/item/{story_id}.json"
        story_data = requests.get(story_url).json()

        # Check if the story's time is less then the cutoff. If that's the case,
        #   we can break early, since we know that any stories fetched from now
        #   on will have a 'time' less than the cutoff as well, since the
        #   'newstories.json' endpoint sorts stories in desc chronological order.
        if story_data['time'] < cutoff_timestamp:
            print("Reached cutoff time.")
            break

        if max_stories is not None and len(stories) == max_stories:
            print(f"Reached '{max_stories}' max_stories.")
            break

        # Skip this story if it doesn't have an URL.
        if 'url' not in story_data:
            continue

        # Try fetching and extract text from the URL. If HTTPError is raised, skip.
        try:
            story_data['text'] = extract_text_from_url(story_data['url'])
        except HTTPError:
            print(f"WARNING: HTTPError for story_id: {story_id}")
            continue

        # Include only stories that have at least some characters.
        if len(story_data['text']) < min_chars_per_story:
            print(f"WARNING: Min char reqs not met for story_id: {story_id}")
            continue

        # Add datetime field based on timestamp.
        story_data["dt"] = datetime.fromtimestamp(story_data["time"], tz=tz_).isoformat()

        stories.append(story_data)

        # Checkpointer.
        if len(stories) % 1 == 0:
            print(f"Stories fetched so far: {len(stories)}")

    print(f"Total stories fetched: {len(stories)}")
    print(f"Runtime: {datetime.now() - start_dt}")

    return stories


if __name__ == '__main__':

    max_stories = 10
    stories_cutoff_in_days = 3
    min_chars_per_story = 100

    hacker_news_stories = fetch_hacker_news_stories(
        max_stories=max_stories,
        stories_cutoff_in_days=stories_cutoff_in_days,
        min_chars_per_story=min_chars_per_story,
    )

    file_path = 'hacker_news_stories.json'
    with open(file_path, 'w') as json_file:
        # noinspection PyTypeChecker
        json.dump(hacker_news_stories, json_file)
    print(f"Stories written to {file_path}")
