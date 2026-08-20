# M3 本机性能记录

测试日期：2026-08-17。机器为双路 Intel Xeon Gold 6148（80 逻辑 CPU），Linux 5.4；ONNX Runtime 1.20.1 CPU。使用同一 PPH Bundle，预热 100 次，每组执行 1000 次单样本 localhost HTTP 请求。结果受共享机器负载与每请求新建连接影响，只用于方案比较和 Worker 选择，不作为 CI 门槛。

| 路径 | 并发/Worker | p50 ms | p95 ms | p99 ms | QPS | RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Python Adapter | 1 | 2.7008 | 2.7499 | 2.7739 | 369.55 | 220.07 |
| C++ ONNX 顺序 | 1 / 2 | 2.3966 | 2.4710 | 2.4897 | 415.89 | 28.80 |
| C++ ONNX 并发 | 1 / 1 | 2.3981 | 2.4767 | 2.5091 | 414.27 | 29.05 |
| C++ ONNX 并发 | 2 / 2 | 2.9242 | 3.2094 | 3.3372 | 656.32 | 29.29 |
| C++ ONNX 并发 | 4 / 4 | 3.9393 | 4.9464 | 5.5268 | 978.54 | 28.93 |

本轮严格使用 `onnx`，fallback 次数为 0。localhost 建连和 HTTP/JSON 占了明显比例，因此不能把这里的延迟当作纯模型耗时。可重复运行：

```bash
PYTHONPATH="$PWD/PythonServices/TreeSemModelAdapter" \
"${TREESEM_MODEL_PYTHON:-python3}" \
  tools/benchmark_treesem_serving.py \
  --server build/treesem_server \
  --bundle artifacts/treesem/pph/pph-seed42-1a299a474ce5 \
  --warmup 100 --requests 1000
```
