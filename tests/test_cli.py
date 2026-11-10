import json

from ttucs.cli import build_parser, main


def test_parse_command(capsys):
    assert main(["parse", "project nightfall, glass smash, take two"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["project"] == "Nightfall" and out["take"] == "2"


def test_help_lists_commands():
    text = build_parser().format_help()
    for cmd in ("generate-data", "train", "eval", "predict", "parse", "report", "info",
                "download", "serve"):
        assert cmd in text


def test_serve_parser_defaults():
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1" and args.port == 0 and args.top_k == 5
    assert args.no_token is False and args.no_warm is False and args.token_file is None
