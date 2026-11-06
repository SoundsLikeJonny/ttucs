# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-06

### Added
- UCS v8.2.1 list (753 CatIDs, 82 categories) shipped as package data with CatID/alias lookup,
  UCS filename building and parsing.
- Rule-based speech-to-text slate parser (project, location, creator, date, microphones, mic setup,
  recorder, sample rate, bit depth, take/scene/slate, perspective, notes, description).
- LLM-free balanced synthetic description and slate generators; BOOM ONE metadata ingestion
  (`ttucs fetch-boom`, local use only); group-based deterministic train/val/test splits.
- Models: hashed n-gram bag-of-embeddings (default, CPU), frozen MiniLM encoder + head, zero-shot
  label-document similarity.
- `TTUCS` Python API: top-k, hierarchical category scores, temperature calibration, unknown /
  category-MISC fallback, UCS filename + FXName + keyword suggestions, JSON serialisation.
- MLflow tracking: params, per-epoch and per-split metrics, per-class reports, confusion artifacts,
  pyfunc model with signature, model registry.
- CLI `ttucs generate-data|train|eval|predict|parse|fetch-boom|report`, YAML configs, tests, CI,
  pre-commit, Makefile, Dockerfile, model and data cards.

### Removed
- Legacy prototype (`main_web.py`, notebook, `.idea/`, `requirements.txt`).

[Unreleased]: https://github.com/SoundsLikeJonny/ttucs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SoundsLikeJonny/ttucs/releases/tag/v0.1.0
