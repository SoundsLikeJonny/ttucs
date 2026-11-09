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


def test_catid_phrases_reference_real_classes(taxonomy):
    gen = DescriptionGenerator(taxonomy, seed=0)
    known = {e.cat_id for e in taxonomy.entries}
    assert len(gen.phrases) > 300
    for cid, phrases in gen.phrases.items():
        assert cid in known, cid
        assert phrases and all(p.strip() and len(p.split()) <= 5 for p in phrases), cid


def test_category_objects_route_to_specific_subcategories(taxonomy):
    gen = DescriptionGenerator(taxonomy, seed=0)
    assert "lawn mower" in gen._routed["MACHGrdn"]
    assert "gate" in gen._routed["DOORGate"]
    # MISC keeps only objects no specific sibling claims
    assert "lawn mower" not in gen._unrouted["MACHINES"]
    misc = " ".join(gen.generate(taxonomy["MACHMisc"], 60))
    assert "lawn mower" not in misc and "elevator" not in misc


def test_phrase_prob_controls_phrase_usage(taxonomy):
    e = taxonomy["MACHGrdn"]
    off = DescriptionGenerator(taxonomy, seed=2, phrase_prob=0.0).generate(e, 80)
    on = DescriptionGenerator(taxonomy, seed=2, phrase_prob=1.0).generate(e, 80)
    assert not any("weed whacker" in t for t in off)
    assert any("weed whacker" in t or "leaf blower" in t for t in on)


def test_verbose_prob_controls_length(taxonomy):
    e = taxonomy["DOORCreak"]
    off = DescriptionGenerator(taxonomy, seed=7, verbose_prob=0.0).generate(e, 60)
    on = DescriptionGenerator(taxonomy, seed=7, verbose_prob=1.0).generate(e, 60)

    def avg(ts):
        return sum(len(t.split()) for t in ts) / len(ts)

    assert avg(on) > avg(off) + 1


def test_verbose_ambience_is_scene_observation(taxonomy):
    e = taxonomy["AMBTraf"]
    texts = DescriptionGenerator(taxonomy, seed=7, verbose_prob=1.0).generate(e, 40)
    openers = ("standing", "you", "recorded", "ambience")
    assert all(t.split(",")[0].split()[0] in openers for t in texts)
    assert all("traffic" in t for t in texts)


def test_verbose_keeps_class_nouns(taxonomy):
    gen = DescriptionGenerator(taxonomy, seed=7, verbose_prob=1.0)
    dog = " ".join(gen.generate(taxonomy["ANMLDog"], 60))
    assert "moo" not in dog and "oink" not in dog and "whinny" not in dog
    assert "elephant" not in dog


def test_multi_clause_joins_same_class_pairs(taxonomy):
    e = taxonomy["MACHGrdn"]
    off = DescriptionGenerator(taxonomy, seed=5, multi_clause_prob=0.0).generate(e, 60)
    on = DescriptionGenerator(taxonomy, seed=5, multi_clause_prob=1.0).generate(e, 60)
    assert not any(", " in t for t in off)
    joined = [t for t in on if ", " in t]
    assert len(joined) > 10
    # a join is two full same-class descriptions
    for t in joined[:5]:
        a, b = t.split(", ", 1)
        assert a and b


def test_explanation_fragments_skip_cross_references(taxonomy):
    gen = DescriptionGenerator(taxonomy, seed=0)
    for _ in range(50):
        frag = gen._expl_fragment(taxonomy["MACHGrdn"]) or ""
        assert "tools-garden" not in frag and "see also" not in frag


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
