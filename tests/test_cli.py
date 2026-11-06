import json

from ttucs.cli import build_parser, main


def test_parse_command(capsys):
    assert main(["parse", "project nightfall, glass smash, take two"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["project"] == "Nightfall" and out["take"] == "2"


def test_help_lists_commands():
    text = build_parser().format_help()
    for cmd in ("generate-data", "train", "eval", "predict", "parse", "fetch-boom", "report"):
        assert cmd in text
