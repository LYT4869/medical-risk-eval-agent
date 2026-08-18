FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && groupadd --system treesem && \
    useradd --system --gid treesem --home /app treesem
WORKDIR /app
COPY PythonServices/TreeSemKnowledge/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir torch==2.5.1+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY PythonServices/TreeSemKnowledge /app
RUN chown -R treesem:treesem /app
USER treesem
EXPOSE 8092
CMD ["python", "-m", "knowledge.server"]
