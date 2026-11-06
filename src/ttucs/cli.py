"""Command line interface: ``ttucs generate-data|train|eval|predict|parse|fetch-boom|report``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ttucs._version import __version__

DEFAULT_MODEL = "models/ttucs-bow"


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", default=None, help="YAML config (defaults are used if omitted)")
    p.add_argument("-o", "--override", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted config override, e.g. -o train.epochs=3 (repeatable)")


def _texts(args: argparse.Namespace) -> list[str]:
    texts = list(args.text or [])
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            texts += [line.strip() for line in fh if line.strip()]
    if not texts and not sys.stdin.isatty():
        texts = [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        raise SystemExit("no input text: pass TEXT arguments, --file or pipe lines on stdin")
    return texts


# ----------------------------------------------------------------------------- commands
def cmd_generate_data(args: argparse.Namespace) -> int:
    from ttucs.config import load_config
    from ttucs.train import generate_data

    cfg = load_config(args.config, args.override)
    splits = generate_data(cfg, args.out)
    for k, v in splits.items():
        print(f"{k:<20} {len(v):>8} examples  {len({e.label for e in v}):>4} labels")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from ttucs.config import load_config
    from ttucs.train import train

    cfg = load_config(args.config, args.override)
    summary = train(cfg, rebuild_data=args.rebuild_data)
    print(json.dumps({k: summary.get(k) for k in ("output_dir", "temperature", "mlflow_run_id",
                                                     "model_uri", "registered_model_version")},
                     indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ttucs.config import load_config
    from ttucs.datasets.build import read_jsonl
    from ttucs.evaluate import (
        SLATE_SPLITS,
        evaluate_split,
        format_table,
        parser_field_accuracy,
        predictions_csv,
    )
    from ttucs.pipeline import TTUCS

    cfg = load_config(args.config, args.override)
    pipe = TTUCS.from_pretrained(args.model)
    splits: dict = {}
    if args.data:
        for path in args.data:
            splits[Path(path).stem] = read_jsonl(path)
    else:
        proc = Path(cfg["data"].get("processed_dir", "data/processed"))
        for f in sorted(proc.glob("test*.jsonl")):
            splits[f.stem] = read_jsonl(f)
        rs = cfg["data"].get("realistic_slates")
        if rs and Path(rs).is_file():
            splits["test_realistic"] = read_jsonl(rs)
    if not splits:
        raise SystemExit("no evaluation data found: run `ttucs generate-data` or pass --data FILE")
    metrics = {}
    for name, exs in splits.items():
        parse = args.parse or name in SLATE_SPLITS or any("slate" in e.source for e in exs[:5])
        m, _ = evaluate_split(pipe, exs, parse=parse)
        if m:
            metrics[name] = m
    print(format_table(metrics))
    parser_m = parser_field_accuracy(splits.get("test_slates_synth", []), pipe.parser)
    if parser_m:
        print("\nslate parser (synthetic slates):")
        for k, v in parser_m.items():
            print(f"  {k:<34} {v:.4f}")
    if args.predictions and "test_realistic" in splits:
        predictions_csv(pipe, splits["test_realistic"], args.predictions)
        print(f"\nwrote {args.predictions}")
    if args.json:
        Path(args.json).write_text(json.dumps({"metrics": metrics, "parser": parser_m}, indent=2))
    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=f"eval:{Path(args.model).name}"):
            mlflow.set_tag("stage", "eval")
            mlflow.log_param("model_dir", args.model)
            for split, m in metrics.items():
                mlflow.log_metrics({f"{split}.{k}": float(v) for k, v in m.items()})
            if parser_m:
                mlflow.log_metrics(parser_m)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from ttucs.pipeline import TTUCS

    pipe = TTUCS.from_pretrained(args.model, **({"unknown_threshold": args.threshold}
                                                if args.threshold is not None else {}))
    preds = pipe.predict_batch(_texts(args), top_k=args.top_k, creator_id=args.creator_id,
                               source_id=args.source_id)
    if args.json:
        out = [p.to_dict() for p in preds]
        print(json.dumps(out if len(out) > 1 else out[0], indent=2, ensure_ascii=False))
        return 0
    for p in preds:
        print(f"\n> {p.text}")
        print(f"  description : {p.description}")
        print(f"  prediction  : {p.cat_id}  ({p.category_full})  conf {p.confidence:.3f}  [{p.decision}]")
        print("  top-k       : " + ", ".join(f"{a.cat_id} {a.confidence:.3f}" for a in p.alternatives))
        print(f"  filename    : {p.filename}")
        print(f"  keywords    : {', '.join(p.keywords)}")
        meta = {k: v for k, v in p.slate.items()
                if v and k not in {"raw_text", "normalized_text", "description", "unparsed"}}
        if meta:
            print(f"  slate       : {json.dumps(meta, ensure_ascii=False)}")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from ttucs.slate import parse_slate

    out = [parse_slate(t).to_dict() for t in _texts(args)]
    if not args.verbose:
        out = [{k: v for k, v in d.items() if v} for d in out]
    print(json.dumps(out if len(out) > 1 else out[0], indent=2, ensure_ascii=False))
    return 0


def cmd_fetch_boom(args: argparse.Namespace) -> int:
    from ttucs.datasets.boom import fetch_boom_one

    path = fetch_boom_one(args.dest)
    print(f"saved BOOM ONE metadata to {path} (c) BOOM Library - do not redistribute")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ttucs.config import load_config
    from ttucs.reporting import distribution_report
    from ttucs.taxonomy import UCSTaxonomy
    from ttucs.train import load_or_build

    cfg = load_config(args.config, args.override)
    splits = load_or_build(cfg)
    tax = UCSTaxonomy.load(cfg["ucs_version"])
    path = distribution_report(splits["train"], tax, args.out, cfg["data"]["boom"].get("path"),
                               set(cfg["data"].get("exclude_categories") or []))
    print(f"wrote {path}")
    return 0


# ------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ttucs", description="Text to UCS: classify spoken slates "
                                "into Universal Category System CatIDs.")
    p.add_argument("--version", action="version", version=f"ttucs {__version__}")
    p.add_argument("-v", "--verbose-log", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("generate-data", help="build synthetic + real train/val/test splits")
    _add_config(s)
    s.add_argument("--out", default=None, help="output dir (default: data.processed_dir)")
    s.set_defaults(func=cmd_generate_data)

    s = sub.add_parser("train", help="train, calibrate, evaluate and log to MLflow")
    _add_config(s)
    s.add_argument("--rebuild-data", action="store_true", help="ignore cached generated data")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="evaluate a trained model")
    _add_config(s)
    s.add_argument("-m", "--model", default=DEFAULT_MODEL)
    s.add_argument("--data", nargs="*", help="JSONL files with text/label (default: test splits)")
    s.add_argument("--parse", action="store_true", help="run the slate parser on --data texts")
    s.add_argument("--predictions", default=None, help="write realistic-slate predictions CSV")
    s.add_argument("--json", default=None, help="write metrics JSON")
    s.add_argument("--mlflow", action="store_true", help="log an evaluation run to MLflow")
    s.set_defaults(func=cmd_eval)

    for name, func, hlp in (("predict", cmd_predict, "classify slate transcripts"),
                            ("parse", cmd_parse, "parse slate transcripts (no model needed)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("text", nargs="*")
        s.add_argument("-f", "--file", help="read one transcript per line")
        if name == "predict":
            s.add_argument("-m", "--model", default=DEFAULT_MODEL)
            s.add_argument("-k", "--top-k", type=int, default=5)
            s.add_argument("--threshold", type=float, default=None, help="unknown threshold")
            s.add_argument("--creator-id", default=None)
            s.add_argument("--source-id", default=None)
            s.add_argument("--json", action="store_true")
        else:
            s.add_argument("--verbose", action="store_true", help="include empty fields")
        s.set_defaults(func=func)

    s = sub.add_parser("fetch-boom", help="download BOOM ONE metadata (local research use)")
    s.add_argument("--dest", default="data/raw/boom_one.csv")
    s.set_defaults(func=cmd_fetch_boom)

    s = sub.add_parser("report", help="class distribution report + plots")
    _add_config(s)
    s.add_argument("--out", default="docs/assets")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose_log else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("alembic", "urllib3", "matplotlib", "PIL", "sentence_transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
