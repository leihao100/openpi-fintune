import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer

tok = PaligemmaTokenizer(max_len=48)
tokens, mask = tok.tokenize("Put the pink cylinder into the orange box.")

print(f"?? token ?: {int(np.sum(mask))} / 48")
print(f"?? token: {tokens[mask]}")  # ??? tokenize ???