from collections import Counter

from ttucs.config import load_config
from ttucs.datasets.build import build_datasets, group_split, label_documents, read_jsonl, write_jsonl
from ttucs.slate import parse_slate
from ttucs.synth.generator import DescriptionGenerator, Example, SlateGenerator


def test_generator_is_deterministic(taxonomy):
    e = taxonomy["DOORWood"]
    a = DescriptionGenerator(taxonomy, seed=1).generate(e, 20)
    b = DescriptionGenerator(taxonomy, seed=1).generate(e, 20)
    assert a == b and len(set(a)) == len(a) == 20


def test_generator_covers_every_trainable_class(taxonomy):
    gen = DescriptionGenerator(taxonomy, seed=0)
    for e in taxonomy.trainable():
        out = gen.generate(e, 3)
        assert out and all(t.strip() for t in out), e.cat_id


def test_slate_generator_truth_is_recoverable():
    sg = SlateGenerator(seed=3)
    hits = 0
    for _ in range(50):
        text, truth = sg.wrap("metal gate slam")
        assert truth["description"] == "metal gate slam"
        hits += "metal gate slam" in parse_slate(text).description
    assert hits >= 40


def test_group_split_keeps_groups_together():
    exs = [Example(text=f"t{i}", label="X", source="s", group=f"g{i // 5}") for i in range(500)]
    tr, va, te = group_split(exs, 0.1, 0.1, seed=0)
    assert len(tr) + len(va) + len(te) == 500
    groups = [{e.group for e in s} for s in (tr, va, te)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


def test_label_documents(taxonomy):
    docs = label_documents(taxonomy, ["DOORWood"])
    assert docs and all(d.label == "DOORWood" and d.source == "ucs_doc" for d in docs)


def test_jsonl_roundtrip(tmp_path):
    exs = [Example("a door", "DOORWood", "synthetic", "g", {"k": 1})]
    assert read_jsonl(write_jsonl(exs, tmp_path / "x.jsonl"))[0] == exs[0]


def test_build_datasets_balances_classes(taxonomy, tmp_path):
    cfg = load_config("configs/smoke.yaml", [f"data.processed_dir={tmp_path}",
                                             "data.realistic_slates=null"])
    splits = build_datasets(cfg, taxonomy)
    counts = Counter(e.label for e in splits["train"])
    assert len(counts) == 738
    assert min(counts.values()) >= 1
    assert {"train", "val", "test_synth", "test_slates_synth"} <= set(splits)
