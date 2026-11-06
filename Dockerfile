FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install --no-cache-dir -e ".[mlflow]"
ENTRYPOINT ["ttucs"]
CMD ["--help"]
