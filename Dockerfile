FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
# Weights: build them into packages/ttucs-model-bow first (scripts/build_model_package.py), or drop
# the local package line once ttucs-model-bow is on PyPI.
RUN pip install --no-cache-dir -e packages/ttucs-model-bow -e ".[train]"
ENTRYPOINT ["ttucs"]
CMD ["--help"]
