import json
with open("tier0/wikipedia_synopses.txt",'w') as out:
    with open("tier0/wikipedia_synopses.json") as f:
        j = json.loads(f.read())
        for i,season in enumerate(j):
            for episode in season:
                out.write("Season "+str(i+1)+" Episode "+
                episode["subseason_ep_number"]+"\n")
                out.write(episode["title"]+"\n")
                out.write(episode["synopsis"]+"\n\n")

with open("tier0/fim_wiki_summaries.txt",'w') as out:
    with open("tier0/fim_wiki_summaries.json") as f:
        j2 = json.loads(f.read())
        for i,season in enumerate(j2):
            for i2,episode in enumerate(season["eps"]):
                out.write("Season "+str(i+1)+" Episode "+str(i2+1)+"\n")
                out.write(j[i][i2]["title"]+"\n")
                out.write(episode["summary"]+"\n\n")

