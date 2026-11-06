import pytest

from ttucs.slate import parse_slate

EXAMPLE = ("Project Thick As Thieves, location Squamish forest, recording a wooden door creak slow, "
           "Sennheiser MKH 416 and Sound Devices MixPre, 96k 24 bit, take 3")


def test_reference_slate():
    s = parse_slate(EXAMPLE)
    assert s.description == "wooden door creak slow"
    assert s.project == "Thick As Thieves"
    assert s.location == "Squamish Forest"
    assert s.microphones == ["Sennheiser MKH 416"]
    assert s.recorder and s.recorder.startswith("Sound Devices MixPre")
    assert (s.sample_rate, s.bit_depth, s.take) == (96000, "24", "3")


def test_unpunctuated_spoken_slate():
    s = parse_slate("um okay project nightfall we are at tofino beach waves crashing on rocks "
                    "mkh eighty forty ortf into a mix pre six ninety six k twenty four bit take eleven")
    assert s.project == "Nightfall"
    assert s.location == "Tofino Beach"
    assert s.description.startswith("waves crashing on rocks")
    assert s.microphones == ["Sennheiser MKH 8040"]
    assert s.mic_setup == ["ORTF"]
    assert s.recorder == "Sound Devices MixPre-6"
    assert (s.sample_rate, s.bit_depth, s.take) == (96000, "24", "11")


def test_decimal_sample_rate_and_perspective():
    s = parse_slate("this is forty four one k sixteen bit zoom h six, car door slam close perspective, "
                    "take two")
    assert s.sample_rate == 44100 and s.bit_depth == "16" and s.recorder == "Zoom H6"
    assert s.description == "car door slam"
    assert "CU" in s.perspective


def test_creator_date_notes():
    s = parse_slate("recorded by jon evans november sixth twenty twenty six, glass bottle shatter on "
                    "concrete, notes: a little wind noise")
    assert s.creator == "Jon Evans"
    assert s.date == "2026-11-06"
    assert s.description == "glass bottle shatter on concrete"
    assert s.notes and "wind" in s.notes


def test_plain_description_passthrough():
    s = parse_slate("dog barking in the distance")
    assert "dog barking" in s.description
    assert s.project is None and s.take is None


@pytest.mark.parametrize("text", ["", "   ", "um uh"])
def test_degenerate_inputs(text):
    s = parse_slate(text)
    assert isinstance(s.description, str)


def test_user_data_from_gear():
    s = parse_slate("mkh eighty forty ortf, birds chirping")
    assert s.user_data == "MKH8040-ORTF"
    assert s.to_dict()["mic_setup"] == ["ORTF"]
