# treeSem backend：本地构建

当前默认构建 HTTP 核心、treeSem 服务入口、ONNX Runtime 和独立的 treeSem MySQL 模块；不构建原项目五子棋示例，也不复用其连接池。

## 1. 构建 Muduo 网络核心

本机 Protobuf 版本较新，而 Muduo 自带的旧 RPC 示例与其不兼容。treeSem 当前不依赖 Muduo RPC，因此在配置 Muduo 时禁用 Protobuf 包即可：

```bash
cmake -S ../deps/muduo -B ../deps/muduo-core-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUDUO_BUILD_EXAMPLES=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_Protobuf=TRUE \
  -DCMAKE_INSTALL_PREFIX="$PWD/../deps/muduo-core-install"
cmake --build ../deps/muduo-core-build -j2
cmake --install ../deps/muduo-core-build
```

## 2. 安装固定 ONNX 依赖

Python 导出和黄金验证使用：

```bash
/home/data/liyingting/miniconda3/envs/triVae/bin/python -m pip install \
  onnx==1.17.0 onnxruntime==1.20.1
```

C++ 使用官方 CPU 预编译包，不把依赖二进制提交仓库：

```bash
curl -fL -o ../deps/onnxruntime-linux-x64-1.20.1.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
echo '67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105  ../deps/onnxruntime-linux-x64-1.20.1.tgz' | sha256sum -c -
tar -xzf ../deps/onnxruntime-linux-x64-1.20.1.tgz -C ../deps
```

## 3. 构建和测试 treeSem 后端

先安装 MySQL Connector/C++ JDBC 兼容包：

```bash
sudo apt-get install libmysqlcppconn-dev libargon2-dev
```

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMUDUO_ROOT="$PWD/../deps/muduo-core-install" \
  -DNLOHMANN_JSON_ROOT="/home/data/liyingting/miniconda3" \
  -DKAMA_ENABLE_ONNXRUNTIME=ON \
  -DKAMA_ENABLE_MYSQL=ON \
  -DKAMA_ENABLE_AUTH=ON \
  -DONNXRUNTIME_ROOT="$PWD/../deps/onnxruntime-linux-x64-1.20.1"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

`-DKAMA_ENABLE_ONNXRUNTIME=OFF` 可构建 remote-only 兼容版本；运行时必须设置 `TREESEM_MODEL_BACKEND=remote`。

`-DKAMA_ENABLE_MYSQL=OFF` 可生成不链接 Connector 的开发构建，但运行时只允许 `TREESEM_STORAGE_BACKEND=memory`。正式默认仍为 MySQL。

## 4. 导出 M2 Serving Bundle

导出只执行一次。它从可信 PT 产物和原始 CSV 重建训练期划分与 `StandardScaler`，随后 Serving 不再读取 CSV。下面的本地演示 Bundle 包含 1489×49 的派生 reference matrix，不能当作生产包提交或发布：

```bash
PYTHONPATH="$PWD/PythonServices/TreeSemModelAdapter" \
/home/data/liyingting/miniconda3/envs/triVae/bin/python \
  -m treesem_adapter.export_bundle \
  --artifact /home/data/liyingting/models/TRI_VAE/runs/compat_tau_fix_20260801/pph/main/pph_semantic_frontier_maxdepth3_seed42_tau_sharpen_20260801.pt \
  --raw-file /home/data/liyingting/models/TRI_VAE/content/test.csv \
  --output-dir "$PWD/artifacts/treesem/pph" \
  --include-reference-dataset
```

不带 `--include-reference-dataset` 时生成生产形态 Bundle；此时 `sample_index` 会返回 400，命名原始输入和预处理输入仍可用。生成目录已被 `.gitignore` 排除。

## 5. 导出 M3 ONNX 子图

```bash
PYTHONPATH="$PWD/PythonServices/TreeSemModelAdapter" \
/home/data/liyingting/miniconda3/envs/triVae/bin/python \
  -m treesem_adapter.export_onnx \
  --bundle "$PWD/artifacts/treesem/pph/pph-seed42-1a299a474ce5"
```

导出器固定 opset 17，运行 `onnx.checker`、shape inference，并在 1489 个 reference 样本上比较 PyTorch 和 Python ONNX Runtime。成功后才把 `model.onnx` 与 checksum 写入 manifest。

生产 Bundle 默认不含 reference matrix。对它导出时，用同版本的本地验证 Bundle 提供校验数据：`--bundle /production/bundle --validation-bundle /local/reference/bundle`；reference 数据不会被复制到生产 Bundle。

## 6. 启动 treeSem 模型 Adapter

Adapter 是常驻 Python 进程：启动时加载并校验一次 Bundle，请求期间不重复加载。Bundle 模式不依赖研究源码、原始 CSV、sklearn 模型对象或训练逻辑。

```bash
PYTHONPATH="$PWD/PythonServices/TreeSemModelAdapter" \
/home/data/liyingting/miniconda3/envs/triVae/bin/python \
  -m treesem_adapter.server \
  --bundle "$PWD/artifacts/treesem/pph/pph-seed42-1a299a474ce5" \
  --port 18081
```

另开终端检查 Adapter：

```bash
curl -i http://127.0.0.1:18081/health
curl -i -X POST http://127.0.0.1:18081/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"sample_index":0}'
```

历史 `--artifact + --code-root + --raw-file` 启动方式仅保留为迁移兼容入口，不作为默认运行方式。

## 7. 启动 C++ ONNX 默认主链

```bash
TREESEM_MODEL_BACKEND=onnx_fallback \
TREESEM_SERVING_BUNDLE_DIR="$PWD/artifacts/treesem/pph/pph-seed42-1a299a474ce5" \
TREESEM_STORAGE_BACKEND=memory \
./build/treesem_server 18080
curl -i http://127.0.0.1:18080/health
curl -i -X POST http://127.0.0.1:18080/api/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"sample_index":0}'
```

预期响应体：

```json
{"status":"ok","service":"treeSem-backend","model_version":"pph-seed42-1a299a474ce5","configured_backend":"onnx_fallback","primary_backend":"onnx","fallback_enabled":true}
```

`/internal/v1/*` 与公开接口复用同一业务事实源。required 模式下只能由持有当前 Run Capability 的 Python Agent 调用，裸 Session Header 会被拒绝。

可以通过环境变量覆盖默认配置：

- `TREESEM_MODEL_ADAPTER_URL`：默认 `http://127.0.0.1:18081/v1/predict`
- `TREESEM_MODEL_CONNECT_TIMEOUT_MS`：默认 `500`
- `TREESEM_MODEL_TIMEOUT_MS`：默认 `5000`
- `TREESEM_INFERENCE_WORKERS`：默认 `2`
- `TREESEM_INFERENCE_QUEUE_CAPACITY`：默认 `32`
- `TREESEM_MODEL_BACKEND`：默认 `onnx_fallback`，还支持 `remote`、`onnx`、`shadow`
- `TREESEM_SERVING_BUNDLE_DIR`：ONNX 相关模式必填
- `TREESEM_STORAGE_BACKEND`：默认 `mysql`；单元测试/显式开发可设 `memory`
- `TREESEM_DATABASE_WORKERS` / `TREESEM_DATABASE_QUEUE_CAPACITY`：默认 `4` / `64`
- `TREESEM_SESSION_TTL_SECONDS`：默认 `3600`
- `TREESEM_COOKIE_SECURE`：本地 HTTP 默认 `false`，TLS 环境必须为 `true`

服务新代码与对外名称统一使用 `treeSem`；历史训练代码和已有产物中的 `trivae` 名称保持不变，以免破坏旧模型加载。

## 8. 启动 M4 MySQL 开发环境

```bash
docker compose -f docker-compose.m4.yml up -d
export TREESEM_DB_PASSWORD=treesem_dev_password
./scripts/migrate_treesem_db.sh

TREESEM_MODEL_BACKEND=onnx \
TREESEM_SERVING_BUNDLE_DIR="$PWD/artifacts/treesem/pph/pph-seed42-1a299a474ce5" \
TREESEM_STORAGE_BACKEND=mysql \
TREESEM_DB_PASSWORD="$TREESEM_DB_PASSWORD" \
./build/treesem_server 18080
```

真实凭据只放环境变量或未提交的 `.env`，不能写进日志或仓库。完整 M4 API、测试库和重启持久化测试见 [M4 文档](m4-business-persistence.md)。

## 9. 启动 M5 Python Agent

使用独立 Python 3.10 环境：

```bash
python3.10 -m venv .venv-agent
. .venv-agent/bin/activate
pip install -r PythonServices/TreeSemAgent/requirements.txt

export PYTHONPATH="$PWD/PythonServices/TreeSemAgent"
export TREESEM_AGENT_LLM_BASE_URL=http://127.0.0.1:8000/v1
export TREESEM_AGENT_LLM_MODEL=your-openai-compatible-model
export TREESEM_AGENT_BACKEND_URL=http://127.0.0.1:18080
export TREESEM_AGENT_SERVICE_SECRET='replace-with-a-distinct-32-byte-secret'
uvicorn server:app --host 127.0.0.1 --port 8091
```

Python `/ready` 只探测 C++ Tool Backend，不发起付费 LLM 请求。CI 使用 Fake LLM，不依赖外网。

## 10. required 认证模式

```bash
export TREESEM_AUTH_MODE=required
export TREESEM_DEPLOYMENT_ENV=local
export TREESEM_ACCESS_JWT_SECRET='replace-with-at-least-32-random-bytes'
export TREESEM_CAPABILITY_JWT_SECRET='replace-with-a-different-32-byte-secret'
export TREESEM_AGENT_SERVICE_SECRET='replace-with-a-third-distinct-secret'
export TREESEM_ALLOWED_ORIGINS=http://127.0.0.1:3000
export TREESEM_REFRESH_COOKIE_SECURE=false
```

三种 Secret 必须不同。生产环境还要求 Secure Refresh Cookie 且禁止通配 Origin。旧匿名 E2E 只能显式设置 `TREESEM_AUTH_MODE=development`。

创建首个管理员：

```bash
export TREESEM_BOOTSTRAP_ADMIN_EMAIL=admin@example.com
export TREESEM_BOOTSTRAP_ADMIN_PASSWORD='a-long-bootstrap-password'
export TREESEM_BOOTSTRAP_ADMIN_DISPLAY_NAME='Local Admin'
./build/treesem-admin bootstrap
```

仅数据库中没有 Admin 时允许 bootstrap。完整 Agent 与安全边界分别见 [M5 文档](m5-agent-core.md)和 [M6 文档](m6-security.md)。

## 11. 运行真实 Bundle/ONNX 测试

配置以下可选参数后，CTest 会额外执行确定性重复导出、checksum 篡改、反归一化和三输入等价性测试：

```bash
cmake -S . -B build \
  -DKAMA_ENABLE_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT="$PWD/../deps/onnxruntime-linux-x64-1.20.1" \
  -DTREESEM_MODEL_PYTHON_EXECUTABLE=/home/data/liyingting/miniconda3/envs/triVae/bin/python \
  -DTREESEM_TEST_BUNDLE_DIR="$PWD/artifacts/treesem/pph/pph-seed42-1a299a474ce5" \
  -DTREESEM_TEST_ARTIFACT=/home/data/liyingting/models/TRI_VAE/runs/compat_tau_fix_20260801/pph/main/pph_semantic_frontier_maxdepth3_seed42_tau_sharpen_20260801.pt \
  -DTREESEM_TEST_RAW_FILE=/home/data/liyingting/models/TRI_VAE/content/test.csv
ctest --test-dir build --output-on-failure
```

该配置会额外运行 1489 样本 Python/C++ 全量一致性、C++ 原生树、损坏 Bundle fail-fast、shadow、Adapter 下线和 ONNX 多 Worker 并发测试。
