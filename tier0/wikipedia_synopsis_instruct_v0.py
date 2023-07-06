import instruct_util
import random
import json
# "Dumb" instruct generated questions just inserting the wikipedia synopses
LOAD_FILE = "wikipedia_synopses.json"
SAVE_FILE = "wikipedia_summary_instruct_v0.json"

INSTRUCT_TEMPLATES = [
    "Can you provide a summary of the episode called {episode} from"
    " {mlp}?",
    "What is a synopsis of the episode {episode} from {mlp}",
    "Please give me a summary of the {mlp} episode titled {episode}.",
    "Could you provide a brief overview of the episode named {episode}"
    "from {mlp}?",
    "I'm looking for a summary of the {mlp} episode {episode}."
    " Can you help?",
    "What happened in the episode {episode} from {mlp}?"
    " Can you summarize it?",
    "Provide a synopsis of the {mlp} episode called {episode}."
    "Can you provide a summary of the {ep_th} episode from the {s_th}"
    "season of {mlp}?",
    "What is a synopsis of the episode numbered {ep} from season {s} of "
    "{mlp}?",
    "Could you provide a brief overview of the episode in S{s}E{ep} "
    "of {mlp}?",
    "What happened in the {mlp} episode from season {s}, episode {ep}?"
    " Can you summarize it?",
    "Provide a synopsis of the episode that is the {ep} episode in season"
    " {s} of MLP:FiM."
]
PRE_RESPONSE_TEMPLATES = [
    "Here is a summary for the episode {episode}"
    " of {mlp}, which is the {ep_th} episode of the {s_th} season: ",
    "Of course. I can provide a summary of the episode {episode} from {mlp},"
    "which is the {ep_th} episode of the {s_th} season: " # idk
]

def gen_summary_question(season_num, episode_title, episode_num):
    temp = random.choice(INSTRUCT_TEMPLATES)
    return temp.format(episode = episode_title,
        mlp = instruct_util.choose_mlp(),
        ep_th = instruct_util.choose_nth(int(episode_num)),
        ep = int(episode_num),
        s_th = instruct_util.choose_nth(int(season_num)),
        s = int(season_num))

with open(LOAD_FILE,'r') as f:
    wiki_synopsis_list = json.loads(f.read())

instruct_dataset = []
for s in range(1,4):
    idx = s - 1
    for ep,x in enumerate(wiki_synopsis_list[idx]):
        instruction = gen_summary_question(
            s, x["title"], x["subseason_ep_number"])
        pre_response = random.choice(PRE_RESPONSE_TEMPLATES).format(
            episode = x["title"],
            mlp = instruct_util.choose_mlp(),
            ep_th = instruct_util.int_to_written(
                int(x["subseason_ep_number"])),
            ep = x["subseason_ep_number"],
            s_th = instruct_util.int_to_written(int(s)),
            s = s)
        instruct_dataset.append({"instruction": instruction,
            "input": "", "output": pre_response
            +x["synopsis"]})

with open(SAVE_FILE,'w') as f:
    f.write(json.dumps(instruct_dataset))

# How long is the wiki summary? Does it fit inside context? Yes, it does.
