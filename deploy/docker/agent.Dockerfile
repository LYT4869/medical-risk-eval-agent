FROM python:3.10-slim-bookworm
RUN groupadd --system treesem && \
    useradd --system --gid treesem --home /app treesem
WORKDIR /app
COPY PythonServices/TreeSemAgent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY PythonServices/TreeSemAgent /app
RUN chown -R treesem:treesem /app
USER treesem
EXPOSE 8091
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8091"]
