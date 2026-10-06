FROM python:3.10-slim-bookworm

WORKDIR /app
COPY PythonServices/TreeSemAgent/requirements.txt /tmp/requirements.txt
COPY PythonServices/TreeSemAgent/requirements-routing.txt /tmp/requirements-routing.txt
RUN python -m pip install --no-cache-dir \
      -r /tmp/requirements.txt -r /tmp/requirements-routing.txt

COPY PythonServices/TreeSemAgent /app
ENTRYPOINT ["python", "-m", "evaluation.benchmark_routing_runtime"]
