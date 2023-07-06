SEASON_LINKS = [
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_1)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_2)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_3)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_4)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_5)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_6)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_7)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_8)",
"https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic_(season_9)",
 ]
# point to json produced by collect_index in tier1/fim_wiki.py 
WIKI_EPISODE_INDEX = r"D:\Code\PPPDataset\tier1\wiki_episode_index.json"
SAVE_FILE = "wikipedia_synopses.json"
from bs4 import BeautifulSoup
import requests
import json

with open(WIKI_EPISODE_INDEX,'r') as f:
    wiki_episode_dict = json.loads(f.read())

def process_season(i,url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    starting_elem = next(x for x in soup.find_all('h2')
        if 'Episodes' in x.text)
    episode_table = next(x for x in starting_elem.find_next('table'))
    tds = episode_table.find_all('td')
    def process_td(td_elem):
        td_paras = []
        td_text = td_elem.text
        if "Part 2" in td_text:
            # If episode is a two-parter
            part1, part2 = td_text.split("Part 2:")
            td_paras.append(part1.split("Part 1:")[1].strip())
            td_paras.append(part2.strip())
            return td_paras
        else:
            return [td_text]
    season_synopses = []

    for y in (
            process_td(x) for x in tds if 'description' in x.get('class',[])):
        for z in y:
            season_synopses.append(z.strip())
    eps_index_dict = wiki_episode_dict[i]["eps"]

    return list({"synopsis": synopsis,
        "subseason_ep_number": ep_dict["subseason_ep_number"],
        "absolute_ep_number": ep_dict["absolute_ep_number"],
        "title": ep_dict["title"],
        "writer": ep_dict["writer"],
        "release_date": ep_dict["release_date"]}
        for synopsis,ep_dict in zip(season_synopses, eps_index_dict))

data = []
for i,url in enumerate(SEASON_LINKS):
    data.append(process_season(i,url))
with open(SAVE_FILE,'w') as f:
    f.write(json.dumps(data))
