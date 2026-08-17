# M4 本机性能记录

记录时间：2026-08-17。该数据用于回归与面试说明，不是跨机器 SLA，也不作为 CI 硬门槛。

## 口径

- C++ ONNX backend，2 个 prediction Worker、2 个 database Worker。
- 内存基线使用 M4 `memory` Store；MySQL 数据使用本地隔离 MySQL 8.0.42、8 条连接和 `READ COMMITTED`。
- 单进程本机 loopback，100 次预热，1000 次顺序 HTTP 请求。
- 测量工具：`tests/benchmark_m4.py`。

| 链路 | p50 | p95 | p99 | QPS |
|---|---:|---:|---:|---:|
| ONNX + M4 memory prediction | 4.009 ms | 4.045 ms | 4.087 ms | 240.96 |
| memory `/ready` | 0.448 ms | 0.503 ms | 1.103 ms | 2087.34 |

MySQL 顺序链路：

| 链路 | p50 | p95 | p99 | QPS |
|---|---:|---:|---:|---:|
| ONNX + MySQL prediction | 15.640 ms | 18.539 ms | 20.579 ms | 61.43 |
| MySQL prediction GET | 10.564 ms | 11.984 ms | 12.526 ms | 90.82 |
| MySQL history GET（limit 20） | 6.127 ms | 7.728 ms | 8.726 ms | 155.11 |
| MySQL `/ready` | 1.306 ms | 1.472 ms | 1.489 ms | 700.79 |

Prediction Worker 与客户端并发同为 1、2、4 时：

| Worker/并发 | p50 | p95 | p99 | QPS |
|---:|---:|---:|---:|---:|
| 1 | 15.432 ms | 17.944 ms | 20.258 ms | 62.38 |
| 2 | 15.662 ms | 19.166 ms | 21.456 ms | 121.75 |
| 4 | 16.628 ms | 21.219 ms | 24.470 ms | 221.62 |

4 Worker 压测采样时 C++ Server RSS 约 41.7 MiB、CPU 约 13.1%、线程数 11。该瞬时采样仅用于量级参考，不等同于完整 profiler 报告。

## 如何解释

这组数字只证明基准脚本、完整 HTTP 链和 M4 业务层可以稳定测量。它不能代表 MySQL 版本性能，也不能与 M3 报告直接归因比较，因为两次测试的客户端连接复用和业务语义不同。

MySQL 让顺序预测的 p50 从约 4.0 ms 增至约 15.6 ms，主要新增了 Session 刷新、连接探活和事务往返；并发从 1 增至 4 时 QPS 接近线性提升，但 p99 也从约 20.3 ms 增至约 24.5 ms。绝对数字受本机、短连接客户端和临时数据库影响，重点是容量趋势与有界失败模式。

当前尚未加入正式 metrics exporter，因此连接池等待、SQL 分段耗时和队列拒绝数没有伪造为“已采集”。M9 会补可观测性后再做完整归因压测。
