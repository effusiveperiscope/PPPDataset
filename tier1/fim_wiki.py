from bs4 import BeautifulSoup
import requests
import re
import json

WIKI_ROOT = "https://mlp.fandom.com/"
WIKI_EPISODE_INDEX_URL = ("https://mlp.fandom.com/wiki/"
    "Friendship_is_Magic_animated_media")
INDEX_TABLE_ORDER = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "Friendship is Forever", "Movie", "Specials", "Best Gift Ever shorts",
    "Miscellaneous shorts"]
INDEX_SAVE_FILE = "wiki_episode_index.json"
TRANSCRIPTS_SAVE_FILE = "episode_transcripts.json"

def collect_index():
    ret = []

    episode_index = requests.get(WIKI_EPISODE_INDEX_URL)
    soup = BeautifulSoup(episode_index.text, 'html.parser')
    tables = soup.find_all('table')
    for i,table in enumerate(tables):
        body = table.find('tbody')
        rows = body.find_all('tr')
        eps = []
        for row in rows:
            cells = row.find_all('td')
            text_info = [x.text.strip() for x in cells]
            if not len(text_info):
                continue
            
            ep_nums = re.findall(r'\d+', text_info[0])
            if ep_nums and len(ep_nums) == 2:
                subseason_ep_number = ep_nums[0]
                absolute_ep_number = ep_nums[1]
            elif ep_nums and len(ep_nums) == 1:
                # not part of main show
                subseason_ep_number = None
                absolute_ep_number = ep_nums[0]
            else:
                subseason_ep_number = None
                absolute_ep_number = None

            eps.append({
                "subseason_ep_number": subseason_ep_number,
                "absolute_ep_number": absolute_ep_number,
                "title": text_info[1],
                "writer": text_info[2],
                "release_date": text_info[3],
                "transcript_href": cells[4].find('a')['href'],
                "episode_summary_href": cells[1].find('a')['href']
            })
        ret.append({"season":INDEX_TABLE_ORDER[i], "eps":eps})

    with open(INDEX_SAVE_FILE, 'w') as f:
        f.write(json.dumps(ret))

# Unfinished, moved to Tier 1
def index_to_summaries():
    index = []
    summaries = []
    seasons_to_grab = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    with open(INDEX_SAVE_FILE, 'r') as f:
        index = json.loads(f.read())
    for s in index:
        if s['season'] not in seasons_to_grab:
            continue
        season_save = {"season": s['season'], "eps":[]}
        for e in s['eps']:
            ep_save = {}
            summary_page = requests.get(WIKI_ROOT + e['episode_summary_href'])
            summary_text = ""
            soup = BeautifulSoup(transcript_page.text, 'html.parser')

            headers = soup.find_all(['h2'])
            summary_first_idx = next(i for i,x in enumerate(texts) if x.text
                == "Summary")
            summary_end_idx = summayr_first_idx + 1

            #ep_save['full_transcript'] = transcript_text
            #season_save["eps"].append(ep_save)
        transcripts.append(season_save)

    with open(TRANSCRIPTS_SAVE_FILE, 'w') as f:
        f.write(json.dumps(transcripts))

def index_to_transcripts():
    index = []
    transcripts = []
    seasons_to_grab = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    with open(INDEX_SAVE_FILE, 'r') as f:
        index = json.loads(f.read())
    for s in index:
        if s['season'] not in seasons_to_grab:
            continue
        season_save = {"season": s['season'], "eps":[]}
        for e in s['eps']:
            ep_save = {}
            transcript_page = requests.get(WIKI_ROOT + e['transcript_href'])
            transcript_text = ""
            soup = BeautifulSoup(transcript_page.text, 'html.parser')

            texts = soup.find_all(['dl','table'])

            script_first_idx = next(i for i,x in enumerate(texts) if x.name !=
                'table')
            navbox_idx = next(i for i,x in enumerate(texts) if 'navbox' in
              x.get('class',[]))

            for x in texts[script_first_idx:navbox_idx]:
                transcript_text = transcript_text + '\n' + x.text + '\n'

            ep_save['full_transcript'] = transcript_text
            season_save["eps"].append(ep_save)
        transcripts.append(season_save)

    with open(TRANSCRIPTS_SAVE_FILE, 'w') as f:
        f.write(json.dumps(transcripts))

#index_to_transcripts()
#collect_index()

