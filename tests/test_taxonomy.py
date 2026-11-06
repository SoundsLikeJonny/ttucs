import pytest

from ttucs.taxonomy import UCS_VERSION, UCSFilename, parse_ucs_filename, sanitize_block


def test_ships_full_ucs_list(taxonomy):
    assert taxonomy.version == UCS_VERSION == "8.2.1"
    assert len(taxonomy) == 753
    assert len(taxonomy.categories) == 82
    assert len(set(taxonomy.cat_ids)) == len(taxonomy)
    assert len(taxonomy.top_level) == 82


def test_trainable_excludes_workflow_categories(taxonomy):
    trainable = taxonomy.trainable()
    assert len(trainable) == 738
    assert all(e.category != "ARCHIVED" for e in trainable)


@pytest.mark.parametrize("query", ["DOORWood", "doorwood", "DOORS-WOOD", " DOORWood "])
def test_resolve_variants(taxonomy, query):
    e = taxonomy.resolve(query)
    assert e is not None and e.cat_id == "DOORWood"
    assert (e.category, e.subcategory, e.category_full) == ("DOORS", "WOOD", "DOORS-WOOD")


def test_aliases(taxonomy):
    assert taxonomy["MACHOffi"].cat_id == "MACHOffc"
    assert taxonomy["WOODHandl"].cat_id == "WOODHndl"
    assert "NOPE" not in taxonomy
    with pytest.raises(KeyError):
        taxonomy["NOPE"]


def test_label_document_uses_explanation_and_synonyms(taxonomy):
    e = taxonomy["DOORWood"]
    doc = e.label_document()
    assert doc.startswith("doors wood.")
    assert e.synonyms and e.synonyms[0] in doc


def test_filename_build_minimal():
    fn = UCSFilename(cat_id="DOORWood", fx_name="Front Door Creak", creator_id="JE", source_id="TAT")
    assert fn.build() == "DOORWood_Front Door Creak_JE_TAT"


def test_filename_build_full_and_sanitised():
    fn = UCSFilename(cat_id="DOORWood", fx_name="Door_creak/slow", creator_id="", source_id="",
                     user_category="EXT", vendor_category="Barn", user_data="MKH416", extension="wav")
    assert fn.build() == "DOORWood-EXT_Barn-Door creak slow_NONE_NONE_MKH416.wav"


def test_filename_roundtrip(taxonomy):
    name = "VEHCar-INT_Sedan Pass By_JE_TAT_MKH8040-ORTF.wav"
    fn = parse_ucs_filename(name, taxonomy)
    assert (fn.cat_id, fn.user_category, fn.fx_name, fn.creator_id, fn.source_id, fn.user_data,
            fn.extension) == ("VEHCar", "INT", "Sedan Pass By", "JE", "TAT", "MKH8040-ORTF", "wav")
    assert fn.build() == name


def test_parse_filename_errors(taxonomy):
    with pytest.raises(ValueError):
        parse_ucs_filename("DOORWood_Creak", taxonomy)
    with pytest.raises(ValueError):
        parse_ucs_filename("XXXXNope_Creak_JE_TAT", taxonomy)


def test_sanitize_block_length():
    assert sanitize_block("a_b:c", 10) == "a b c"
    assert len(sanitize_block("word " * 20, 12)) <= 12
