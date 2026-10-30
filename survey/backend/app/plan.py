"""Question-plan generation"""
import random
from typing import Any, Literal

# placeholders constants
N_SAMPLES = 15
N_REPEATS = 2   # n samples shown a second time to measure consistency
QUESTION_ORDER = ["balanced", "best"] #? randomized per participant or fixed?


def make_plan(
        manifest: dict,
        seed: int,
        n_samples: int = N_SAMPLES,
        n_repeats: int = N_REPEATS,
) -> list[dict]:
    """Return list of questions per participant"""
    rng = random.Random(seed)
    # loops through the sample object and extracts a dictionary keyed with sample "id"s
    samples = {s["id"]: s for s in manifest["samples"]}
    if not samples:
        raise ValueError("manifest has no samples")

    chosen = rng.sample(list(samples), min(n_samples, len(samples)))

    # 1. Base session one normal trial per chosen sample
    entries: list[tuple[Any, Literal["normal", "repeat"]]] = [(sid, "normal") for sid in chosen]

    # 2. Repeats subsample somewhere after the first one
    for sid in rng.sample(chosen, min(n_repeats, len(chosen))):
        first = entries.index((sid, "normal"))
        entries.insert(rng.randint(first+1, len(entries)), (sid, "repeat"))

    # 3. turn entries into trials shuffling image variants order
    plan = []
    # noinspection bad-argument-type
    for index, (sid, kind) in enumerate[tuple[Any, Literal["normal", "repeat"]]](entries):
        variants = [v["id"] for v in samples[sid]["variants"]]
        rng.shuffle(variants)
        plan.append({
            "question_index": index,
            "sample_id": sid,
            "kind": kind,
            "variant_order": variants,
        })
    return plan

