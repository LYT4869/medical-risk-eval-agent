FROM python:3.10-slim-bookworm
ARG TREESEM_INSTALL_SEMANTIC_ROUTING=false
RUN if [ "$TREESEM_INSTALL_SEMANTIC_ROUTING" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
      rm -rf /var/lib/apt/lists/*; \
    elif [ "$TREESEM_INSTALL_SEMANTIC_ROUTING" != "false" ]; then \
      echo "TREESEM_INSTALL_SEMANTIC_ROUTING must be true or false" >&2; \
      exit 2; \
    fi && \
    groupadd --system treesem && useradd --system --gid treesem --home /app treesem
WORKDIR /app
COPY PythonServices/TreeSemAgent/requirements.txt /tmp/requirements.txt
COPY PythonServices/TreeSemAgent/requirements-routing.txt /tmp/requirements-routing.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN if [ "$TREESEM_INSTALL_SEMANTIC_ROUTING" = "true" ]; then \
      pip install --no-cache-dir -r /tmp/requirements-routing.txt; \
    elif [ "$TREESEM_INSTALL_SEMANTIC_ROUTING" != "false" ]; then \
      echo "TREESEM_INSTALL_SEMANTIC_ROUTING must be true or false" >&2; \
      exit 2; \
    fi
COPY PythonServices/TreeSemAgent /app
RUN chown -R treesem:treesem /app
USER treesem
EXPOSE 8091
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8091"]
