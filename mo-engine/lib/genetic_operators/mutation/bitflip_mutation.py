import random

def bitflip_mutation(mask: list[int], p: float) -> list[int]:
    """Bit-flip mutation with per-bit probability p."""
    out = mask[:]
    for i in range(len(out)):
        if random.random() < p:
            out[i] = 1 - out[i]
    return out