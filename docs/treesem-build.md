# treeSem backend：本地构建

当前阶段只构建 HTTP 核心和 treeSem 服务入口，不构建原项目的五子棋示例与 MySQL 模块。

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

## 2. 构建和测试 treeSem 后端

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMUDUO_ROOT="$PWD/../deps/muduo-core-install" \
  -DNLOHMANN_JSON_ROOT="/home/data/liyingting/miniconda3"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## 3. 导出 M2 Serving Bundle

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

## 4. 启动 treeSem 模型 Adapter

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

## 5. 启动 C++ 服务并完成端到端验证

```bash
./build/treesem_server 18080
curl -i http://127.0.0.1:18080/health
curl -i -X POST http://127.0.0.1:18080/api/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"sample_index":0}'
```

预期响应体：

```json
{"status":"ok","service":"treeSem-backend"}
```

`/internal/v1/predictions` 与公开预测接口使用同一套模型代理，可供后续 Agent 内部调用。当前阶段只完成 C++ 到 Python 的预测闭环，不包含 Python Agent 回调 C++ 的重入链路。

可以通过环境变量覆盖默认配置：

- `TREESEM_MODEL_ADAPTER_URL`：默认 `http://127.0.0.1:18081/v1/predict`
- `TREESEM_MODEL_CONNECT_TIMEOUT_MS`：默认 `500`
- `TREESEM_MODEL_TIMEOUT_MS`：默认 `5000`
- `TREESEM_INFERENCE_WORKERS`：默认 `2`
- `TREESEM_INFERENCE_QUEUE_CAPACITY`：默认 `32`

服务新代码与对外名称统一使用 `treeSem`；历史训练代码和已有产物中的 `trivae` 名称保持不变，以免破坏旧模型加载。

## 6. 运行真实 Bundle 测试

配置以下可选参数后，CTest 会额外执行确定性重复导出、checksum 篡改、反归一化和三输入等价性测试：

```bash
cmake -S . -B build \
  -DTREESEM_MODEL_PYTHON_EXECUTABLE=/home/data/liyingting/miniconda3/envs/triVae/bin/python \
  -DTREESEM_TEST_BUNDLE_DIR="$PWD/artifacts/treesem/pph/pph-seed42-1a299a474ce5" \
  -DTREESEM_TEST_ARTIFACT=/home/data/liyingting/models/TRI_VAE/runs/compat_tau_fix_20260801/pph/main/pph_semantic_frontier_maxdepth3_seed42_tau_sharpen_20260801.pt \
  -DTREESEM_TEST_RAW_FILE=/home/data/liyingting/models/TRI_VAE/content/test.csv
ctest --test-dir build --output-on-failure
```
