from ttucs.text import (
    abbreviate,
    initials,
    make_fx_name,
    make_keywords,
    split_description,
    strip_take_numbers,
)


def test_split_description_keyword_tail():
    prose, kws = split_description("Wooden door creaks open slowly. Door, Wood, Creak, Open.")
    assert prose == "Wooden door creaks open slowly."
    assert kws == ["Door", "Wood", "Creak", "Open"]


def test_split_description_without_tail():
    assert split_description("A dog barks twice.") == ("A dog barks twice.", [])


def test_fx_name_budget():
    fx = make_fx_name("a very heavy wooden barn door creaking open slowly in the wind")
    assert len(fx) <= 25 and fx.startswith("Heavy Wooden")


def test_keywords_unique():
    kws = make_keywords("door creak door slam", ["Wood", "Doors"])
    assert kws == ["Door", "Creak", "Slam", "Wood", "Doors"]


def test_ids():
    assert initials("Jon Evans") == "JE"
    assert abbreviate("Thick As Thieves") == "ThickAsThieves"
    assert abbreviate("A Very Long Project Name Here") == "AVLPNH"
    assert strip_take_numbers("Augusta A109 Fly Away 01") == "Augusta A109 Fly Away"
