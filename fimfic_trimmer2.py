from tqdm import tqdm
IN_FILE = "fimficOmegaV3.txt"
OUT_FILE = "fimficOmega_Trimmed.txt"
TARGET_SIZE = 220 * 1024 * 1024 # 200MB dataset
with open(IN_FILE, encoding='utf-8') as f:
    with open(OUT_FILE,'w', encoding='utf-8') as of:
        for l in tqdm(f):
            if (l.startswith("<|startoftext|>") or 
                l.startswith("***") or
                l.startswith("<|endoftext|>") or
                l.startswith("[tags")):
                continue
            of.write(l)
            if of.tell() > TARGET_SIZE:
                break
