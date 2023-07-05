IN_FILE = "fimficOmegaV3.txt"
OUT_FILE = "fimficOmega_Trimmed.txt"
PARAGRAPH_LIMIT = 512

with open(IN_FILE) as f:
    with open(OUT_FILE,'w') as of:
        para_buffer = []
        para_len = 0
        for l in f:
            if (l.startswith("<|startoftext|>") or 
                l.startswith("***") or
                l.startswith("<|endoftext|>") or
                l.startswith("[tags")):
                continue
            para_buffer.append(l)
            para_len += len(l)

            if para_len > PARAGRAPH_LIMIT and len(para_buffer) > 1:
                for l2 in para_buffer[:-1]:
                    of.write(l2)
                of.write("\n\n")
                para_buffer = [para_buffer[-1]]
                para_len = 0
            elif para_len > PARAGRAPH_LIMIT and len(para_buffer) == 1:
                print("Note:"
                    "Single line exceeded PARAGRAPH_LIMIT:"+para_buffer[0])
                of.write(para_buffer[0])
                of.write("\n\n")
                para_buffer = []
                para_len = 0
