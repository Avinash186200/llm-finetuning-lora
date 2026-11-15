import random

from lorafin.data import DIRECTIONS, FIELDS, METRICS, UNITS, generate_splits, make_example


def test_same_seed_same_data():
    a = generate_splits(7, 20, 5, 5, 5)
    b = generate_splits(7, 20, 5, 5, 5)
    assert a == b


def test_different_seed_different_data():
    assert generate_splits(7, 20, 5, 5, 5) != generate_splits(8, 20, 5, 5, 5)


def test_split_sizes_and_schema():
    s = generate_splits(1, 30, 10, 10, 10)
    assert {k: len(v) for k, v in s.items()} == {"train": 30, "val": 10, "test": 10, "hard": 10}
    for rows in s.values():
        for r in rows:
            t = r["target"]
            assert list(t) == FIELDS
            assert t["metric"] in METRICS
            assert t["unit"] in UNITS[t["metric"]]
            assert t["direction"] in DIRECTIONS
            assert t["entity"] in r["text"]


def test_held_out_entities_not_in_train():
    s = generate_splits(3, 200, 30, 30, 30)
    train_ents = {r["target"]["entity"] for r in s["train"]}
    for name in ("val", "test", "hard"):
        assert not train_ents & {r["target"]["entity"] for r in s[name]}


SUFFIXES = ("Dynamics", "Holdings", "Labs", "Industries", "Group", "Systems", "Foods", "Energy", "Logistics", "Bank")


def test_hard_examples_have_a_distractor():
    rng = random.Random(0)
    for _ in range(100):
        ex = make_example(rng, hard=True)
        two_companies = sum(ex["text"].count(s) for s in SUFFIXES) >= 2
        two_numbers = sum(c.isdigit() for c in ex["text"]) > sum(c.isdigit() for c in str(ex["target"]["value"]))
        assert two_companies or two_numbers
