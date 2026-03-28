import random


def bitflip_mutation(mask: list[int], p: float, rng: random.Random) -> list[int]:
    """Bit-flip mutation with per-bit probability p."""
    out = mask[:]
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = 1 - out[i]
    return out
