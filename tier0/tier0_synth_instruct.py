import requests
import json
import re

# Synthetic dataset using oobabooga api

HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

def fread(p):
    with open(p) as f:
        return f.read()

def jread(p):
    with open(p) as f:
        return json.loads(f.read())

SUMMARY_PROMPT_TEMPLATE = fread("prompt_templates/summary_prompt.txt")
QUESTIONS_PROMPT_TEMPLATE = fread("prompt_templates/questions_prompt.txt")
episode_index = jread(r"D:\Code\PPPDataset\tier1\wiki_episode_index.json")
fim_wiki_summaries = jread("fim_wiki_summaries.json")
wikipedia_synopses = jread("wikipedia_synopses.json")
SAVE_FILE = "tier0_synth_instruct.json"

# Does this use new context?
# Does this not look like broken context to you???
def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 3000,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',  
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 8192,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        print(result)
        return result

def validate_question_response(text):
    questions = []
    answers = []

    pattern = r'\d+\) (.+?)\n(.+?)(?=\n\d+\) |$)'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for match in matches:
        # I don't actually care what the question number is as long as there
        # are six of them
        question, answer = match
        questions.append(question)
        answers.append(answer.strip())

    # Ensure six questions
    if (len(questions) != 7) or (len(answers) != 7):
        return None

    #print("Success")
    return questions, answers

# Seasons
out = []
for i, (epidx, fimwiki, wiki) in enumerate(zip(
    episode_index, fim_wiki_summaries, wikipedia_synopses)):
    s = 1
    season_out = []
    for ep,fw,w in zip(epidx["eps"], fimwiki["eps"], wiki):
        if int(ep["subseason_ep_number"]) != 3: continue
        print("Processing "+ep["title"])
        summary_response = run(SUMMARY_PROMPT_TEMPLATE.format(
            s = s, ep = ep["subseason_ep_number"], synopsis=w["synopsis"],
            summary=fw["summary"]))
        print(QUESTIONS_PROMPT_TEMPLATE.format(
            s = s, ep = ep["subseason_ep_number"], synopsis=w["synopsis"],
            summary=fw["summary"]))
        question_response = validate_question_response(
            run(QUESTIONS_PROMPT_TEMPLATE.format(
            s = s, ep = ep["subseason_ep_number"], synopsis=w["synopsis"],
            summary=fw["summary"])))
        while question_response is None:
            question_response = validate_question_response(
                run(QUESTIONS_PROMPT_TEMPLATE.format(
                s = s, ep = ep["subseason_ep_number"], synopsis=w["synopsis"],
                summary=fw["summary"])))
        questions, answers = question_response
        season_out.append({"summary": summary_response, 
                           "questions": questions,
                           "answers": answers})
    s += 1
    out.append(season_out)

with open(SAVE_FILE,'w') as f:
    f.write(json.dumps(out, indent=4))
