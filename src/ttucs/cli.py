"""Command line interface: ``ttucs predict|parse|info|download|generate-data|train|eval|report``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ttucs._version import __version__

MODEL_HELP = ("model directory or published model name (default: 'bow', resolved offline-first: "
              "installed ttucs-model-bow package -> cache -> download unless TTUCS_OFFLINE=1)")


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
    from ttucs._extras import require_mlflow, require_train_extras

    require_train_extras("`ttucs train`")
    from ttucs.config import load_config
    from ttucs.train import train

    cfg = load_config(args.config, args.override)
    if cfg.get("mlflow", {}).get("enabled", True):
        require_mlflow("`ttucs train` with mlflow.enabled=true")
    summary = train(cfg, rebuild_data=args.rebuild_data)
    print(json.dumps({k: summary.get(k) for k in ("output_dir", "temperature", "mlflow_run_id",
                                                     "model_uri", "registered_model_version")},
                     indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ttucs._extras import require_train_extras

    require_train_extras("`ttucs eval`")
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
    if args.predictions:
        # the realistic slates by default; otherwise the first --data file (e.g. the dev set)
        name = "test_realistic" if "test_realistic" in splits else next(iter(splits))
        predictions_csv(pipe, splits[name], args.predictions)
        print(f"\nwrote {args.predictions} ({name})")
    if args.json:
        Path(args.json).write_text(json.dumps({"metrics": metrics, "parser": parser_m}, indent=2))
    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=f"eval:{Path(str(pipe.model_path)).name}"):
            mlflow.set_tag("stage", "eval")
            mlflow.log_param("model_dir", str(pipe.model_path))
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
        print(json.dumps([p.to_dict() for p in preds], indent=2, ensure_ascii=False))
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


def cmd_report(args: argparse.Namespace) -> int:
    from ttucs._extras import require_train_extras

    require_train_extras("`ttucs report`")
    from ttucs.config import load_config
    from ttucs.reporting import distribution_report
    from ttucs.taxonomy import UCSTaxonomy
    from ttucs.train import load_or_build

    cfg = load_config(args.config, args.override)
    splits = load_or_build(cfg)
    tax = UCSTaxonomy.load(cfg["ucs_version"])
    path = distribution_report(splits["train"], tax, args.out,
                               set(cfg["data"].get("exclude_categories") or []))
    print(f"wrote {path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from ttucs import hub
    from ttucs.taxonomy import UCS_VERSION

    info: dict = {"ttucs_version": __version__, "ucs_version": UCS_VERSION,
                  "offline_mode": hub.is_offline(), "cache_dir": str(hub.cache_root())}
    try:
        resolved = hub.resolve_model(args.model, allow_download=False)
        info["model"] = hub.describe(resolved)
        if args.verify:
            hub.verify_manifest(resolved.path)
            info["model"]["manifest_verified"] = True
    except hub.ModelNotFoundError as exc:
        info["model"] = None
        info["error"] = str(exc)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for k, v in info.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for kk, vv in v.items():
                    print(f"  {kk:<20} {vv}")
            else:
                print(f"{k:<22} {v}")
    return 0 if info["model"] else 1


def cmd_download(args: argparse.Namespace) -> int:
    from ttucs import hub

    path = hub.download_model(args.model, dest=args.dest, url=args.url, sha256=args.sha256, force=args.force)
    print(f"model {args.model!r} ready at {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from ttucs.serve import run_serve

    try:
        return run_serve(args.model, host=args.host, port=args.port, token_file=args.token_file,
                         top_k=args.top_k, warm=not args.no_warm, no_token=args.no_token)
    except ValueError as exc:
        raise SystemExit(f"ttucs serve: error: {exc}") from None


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
    s.add_argument("-m", "--model", default=None, help=MODEL_HELP)
    s.add_argument("--data", nargs="*", help="JSONL files with text/label (default: test splits)")
    s.add_argument("--parse", action="store_true", help="run the slate parser on --data texts")
    s.add_argument("--predictions", default=None,
                   help="write a predictions CSV (realistic slates, else the first --data file)")
    s.add_argument("--json", default=None, help="write metrics JSON")
    s.add_argument("--mlflow", action="store_true", help="log an evaluation run to MLflow")
    s.set_defaults(func=cmd_eval)

    for name, func, hlp in (("predict", cmd_predict, "classify slate transcripts"),
                            ("parse", cmd_parse, "parse slate transcripts (no model needed)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("text", nargs="*")
        s.add_argument("-f", "--file", help="read one transcript per line")
        if name == "predict":
            s.add_argument("-m", "--model", default=None, help=MODEL_HELP)
            s.add_argument("-k", "--top-k", type=int, default=5)
            s.add_argument("--threshold", type=float, default=None, help="unknown threshold")
            s.add_argument("--creator-id", default=None)
            s.add_argument("--source-id", default=None)
            s.add_argument("--json", action="store_true")
        else:
            s.add_argument("--verbose", action="store_true", help="include empty fields")
        s.set_defaults(func=func)

    s = sub.add_parser("info", help="show the resolved model, versions and offline mode (no network)")
    s.add_argument("-m", "--model", default=None, help=MODEL_HELP)
    s.add_argument("--verify", action="store_true", help="check weights against manifest.json sha256")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("download", help="download + verify published weights into the cache")
    s.add_argument("--model", default="bow", help="published model name (default: bow)")
    s.add_argument("--dest", default=None, help="target directory (default: $TTUCS_HOME or user cache)")
    s.add_argument("--url", default=None, help="override the release URL (mirror / local file)")
    s.add_argument("--sha256", default=None, help="expected sha256 of the archive (with --url)")
    s.add_argument("--force", action="store_true", help="re-download even if present")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("serve", help="serve the engine over loopback HTTP for other tools")
    s.add_argument("-m", "--model", default=None, help=MODEL_HELP)
    s.add_argument("--host", default="127.0.0.1", help="loopback address to bind (default: 127.0.0.1)")
    s.add_argument("--port", type=int, default=0, help="port (default: 0 = pick a free one)")
    s.add_argument("--token-file", default=None,
                   help="endpoint file with host/port/token (default: $TTUCS_HOME/serve.json)")
    s.add_argument("--no-token", action="store_true", help="disable the token (tests only)")
    s.add_argument("-k", "--top-k", type=int, default=5, help="default top-k for /predict")
    s.add_argument("--no-warm", action="store_true", help="skip the startup warm-up prediction")
    s.set_defaults(func=cmd_serve)

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
    from ttucs._extras import MissingExtraError
    from ttucs.hub import (
        ChecksumError,
        DownloadError,
        IncompatibleModelError,
        ModelNotFoundError,
        OfflineError,
    )

    try:
        return int(args.func(args) or 0)
    except (MissingExtraError, ModelNotFoundError, OfflineError, ChecksumError,
            IncompatibleModelError, DownloadError) as exc:
        print(f"ttucs {args.command}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
