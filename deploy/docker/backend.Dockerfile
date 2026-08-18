FROM debian:bookworm-slim AS builder

ARG ONNXRUNTIME_VERSION=1.20.1
ARG ONNXRUNTIME_SHA256=67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105
ARG MUDUO_TAG=v2.0.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake curl g++ git make pkg-config \
    libargon2-dev libboost-dev libcurl4-openssl-dev libmysqlcppconn-dev \
    libssl-dev nlohmann-json3-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${MUDUO_TAG}" https://github.com/chenshuo/muduo.git /tmp/muduo && \
    cmake -S /tmp/muduo -B /tmp/muduo/build -DCMAKE_BUILD_TYPE=Release -DMUDUO_BUILD_EXAMPLES=OFF && \
    cmake --build /tmp/muduo/build -j2 && cmake --install /tmp/muduo/build

RUN curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" -o /tmp/onnxruntime.tgz && \
    echo "${ONNXRUNTIME_SHA256}  /tmp/onnxruntime.tgz" | sha256sum -c - && \
    mkdir -p /opt/onnxruntime && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /opt/onnxruntime

WORKDIR /src
COPY . .
RUN cmake -S . -B /tmp/treesem-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DMUDUO_ROOT=/usr/local \
      -DNLOHMANN_JSON_ROOT=/usr \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DKAMA_ENABLE_ONNXRUNTIME=ON \
      -DKAMA_ENABLE_MYSQL=ON \
      -DKAMA_ENABLE_AUTH=ON \
      -DKAMA_BUILD_TESTS=OFF && \
    cmake --build /tmp/treesem-build -j2

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl default-mysql-client libargon2-1 libcurl4 \
    libmysqlcppconn7v5 libssl3 && rm -rf /var/lib/apt/lists/* && \
    groupadd --system treesem && useradd --system --gid treesem --home /app treesem
COPY --from=builder /tmp/treesem-build/treesem_server /usr/local/bin/treesem_server
COPY --from=builder /tmp/treesem-build/treesem-admin /usr/local/bin/treesem-admin
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=builder /src/db /app/db
COPY --from=builder /src/scripts/migrate_treesem_db.sh /app/scripts/migrate_treesem_db.sh
COPY deploy/docker/backend-entrypoint.sh /usr/local/bin/treesem-backend-entrypoint
RUN chmod 0555 /usr/local/bin/treesem-backend-entrypoint /app/scripts/migrate_treesem_db.sh && ldconfig && \
    chown -R treesem:treesem /app
USER treesem
WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["treesem-backend-entrypoint"]
CMD ["treesem_server"]
