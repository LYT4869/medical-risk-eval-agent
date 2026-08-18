FROM python:3.10-slim-bookworm
RUN groupadd --system treesem && useradd --system --gid treesem --home /app treesem
WORKDIR /app
COPY PythonServices/TreeSemModelAdapter/requirements-serving.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir torch==2.5.1+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu
COPY PythonServices/TreeSemModelAdapter /app
RUN chown -R treesem:treesem /app
USER treesem
EXPOSE 18081
CMD ["python", "-m", "treesem_adapter.server", "--bundle", "/models/treesem", "--host", "0.0.0.0", "--port", "18081"]
