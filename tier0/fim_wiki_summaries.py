from bs4 import BeautifulSoup
import requests
import json
from unidecode import unidecode

INDEX_SAVE_FILE = r"D:\Code\PPPDataset\tier1\wiki_episode_index.json"
WIKI_ROOT = "https://mlp.fandom.com/"
SUMMARIES_SAVE_FILE = "fim_wiki_summaries.json"

def index_to_summaries():
    index = []
    summaries = []
    seasons_to_grab = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    #seasons_to_grab = ["1"]
    with open(INDEX_SAVE_FILE, 'r') as f:
        index = json.loads(f.read())
    for s in index:
        if s['season'] not in seasons_to_grab:
            continue
        season_save = {"season": s['season'], "eps":[]}
        for e in s['eps']:
            print("Processing "+e['title'])
            ep_save = {}

            summary = ""
            summary_page = requests.get(WIKI_ROOT + e['episode_summary_href'])
            summary_text = ""
            soup = BeautifulSoup(summary_page.text, 'html.parser')

            headers = soup.find_all(['h2'])
            summary_first_idx = next(i for i,x in enumerate(headers) if 
                "Summary" in x.text)
            summary_end_idx = summary_first_idx + 1

            current_element = headers[summary_first_idx]
            while current_element and current_element != headers[
                summary_end_idx]:
                if current_element.name == 'p':
                    summary += current_element.text
                current_element = current_element.next_sibling

            ep_save['summary'] = unidecode(summary)
            season_save['eps'].append(ep_save)
        summaries.append(season_save)

    with open(SUMMARIES_SAVE_FILE, 'w') as f:
        f.write(json.dumps(summaries))
index_to_summaries()
