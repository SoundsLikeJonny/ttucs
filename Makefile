.PHONY: install lint test data train eval smoke report release-model dist
install: ; pip install -e packages/ttucs-model-bow -e ".[dev]"
lint: ; ruff check src tests scripts
test: ; pytest -q
data: ; ttucs generate-data -c configs/bow.yaml
train: ; ttucs train -c configs/bow.yaml
eval: ; ttucs eval -m models/ttucs-bow --predictions models/ttucs-bow/reports/realistic.csv
smoke: ; ttucs generate-data -c configs/smoke.yaml && ttucs train -c configs/smoke.yaml
report: ; ttucs report -c configs/bow.yaml --out docs/assets
release-model: ; ttucs train -c configs/bow-release.yaml && python scripts/build_model_package.py --model-dir models/ttucs-bow-release --tarball --wheel
dist: ; python -m build --outdir dist . && twine check --strict dist/*
