import random
from tqdm import tqdm
#IN_FILE = "D:/Code/fimfic_800MB.txt"
IN_FILE = "/mnt/nvme1n1p2/Code/fimfic_800MB.txt"
OUT_FILE = "fimfic_800MB_sampled_20MB.txt"
PARAGRAPH_LIMIT = 512
TARGET_SIZE = 20 * 1024 * 1024 # 20MB dataset

with open(IN_FILE,encoding='utf-8') as f:
    output_buffer = []
    para_buffer = []
    para_len = 0
    for l in tqdm(f):
        if (l.startswith("<|startoftext|>") or 
            l.startswith("***") or
            l.startswith("<|endoftext|>") or
            l.startswith("[tags")):
            continue
        para_buffer.append(l)
        para_len += len(l)

        if para_len > PARAGRAPH_LIMIT and len(para_buffer) > 1:
            for l2 in para_buffer[:-1]:
                output_buffer.append(l2)
            para_buffer = [para_buffer[-1]]
            para_len = 0
        elif para_len > PARAGRAPH_LIMIT and len(para_buffer) == 1:
            print("Note:"
                "Single line exceeded PARAGRAPH_LIMIT:"+para_buffer[0])
            output_buffer.append(para_buffer[0])
            para_buffer = []
            para_len = 0

    print("Shuffling...")
    random.shuffle(output_buffer)

    print("Writing output...")
    with open(OUT_FILE,'w',encoding='utf-8') as of:
        for l in tqdm(output_buffer):
            of.write(l)
            of.write("\n\n")
            if of.tell() > TARGET_SIZE:
                break
