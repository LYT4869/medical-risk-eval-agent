FROM python:3.10-slim-bookworm

WORKDIR /app
COPY PythonServices/TreeSemAgent/requirements.txt /tmp/requirements.txt
COPY PythonServices/TreeSemAgent/requirements-routing-export.txt /tmp/requirements-routing-export.txt
RUN python -m pip install --no-cache-dir \
      -r /tmp/requirements.txt -r /tmp/requirements-routing-export.txt

COPY PythonServices/TreeSemAgent /app
ENTRYPOINT ["python", "-m", "tools.export_routing_artifact"]
