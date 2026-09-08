# Agent 混合路由评测报告

评测日期：2026-09-08

## 结论

候选方案满足质量与延迟推广门槛：保留高精度规则作为快路径，规则未命中时使用本地多语言 E5 做语义路由；歧义、组合意图、超时、过载和可选依赖故障统一降级到受 Tool 权限约束的 Open Agent。语义路由只选择工作流，不授予任何新权限。

代价同样明确：语义依赖使 Agent 镜像增加约 1078.1 MiB，进程首次加载模型的 RSS 增量约 745.6 MiB。因此实现保持可选构建，最终是否将 `hybrid_optional` 设为本地演示默认值，要在完整 Agent 回归和定向真实 LLM 回归后决定；资源紧张环境可继续使用 `rule`。

## 可复现配置

- 路由语料：150 条合成、无患者数据用例；held-out 60 条。
- Embedding：`intfloat/multilingual-e5-small`。
- Revision：`614241f622f53c4eeff9890bdc4f31cfecc418b3`。
- 语料 SHA-256：`21f9b16c61b34c74f9b4365e1d545911e48e84eff152c3eb2f2af0717b4e849b`。
- CPU Torch：`2.5.1+cpu`，未包含 CUDA 运行时。
- Executor：1 个 Worker、8 个排队容量、5 ms admission timeout、150 ms route timeout。
- 校准阈值：
  - 最低相似度 `0.8800822593018546`。
  - 最低 Top-1/Top-2 margin `0.002103112350333336`。
  - 次意图相似度上限 `0.9186771161315094`。

阈值仅由 calibration split 选择，held-out 结果不用于反向调参。

## Held-out 结果

| 指标 | 高精度规则 | 规则 + E5 语义回退 |
|---|---:|---:|
| 已知单意图准确率 | 56.67% | 60.00% |
| 业务 Macro F1 | 69.71% | 72.69% |
| 规则精度 | 100% | 100% |
| Unknown 召回率 | 100% | 100% |
| 组合意图回退率 | 100% | 100% |
| 安全用例准确率 | 100% | 100% |
| 路由 p50 | 0.020 ms | 0.092 ms |
| 路由 p95 | 0.027 ms | 18.511 ms |

混合方案提升幅度不大但方向稳定，并保持了“错误确定性路由为零”的首要约束。Embedding 相似度是路由分数，不解释为意图概率。

## 资源证据

测量机器：双路 Intel Xeon Gold 6148，80 个逻辑 CPU；Docker 26.1.3；Linux x86_64。

| 项目 | 实测值 |
|---|---:|
| E5 + 意图样例缓存首次初始化 | 6.329 s |
| 初始化前进程 RSS | 19.2 MiB |
| 初始化后进程 RSS | 764.8 MiB |
| RSS 增量 | 745.6 MiB |
| rule Agent 镜像 | 166.7 MiB |
| semantic Agent 镜像 | 1244.8 MiB |
| 镜像增量 | 1078.1 MiB |

构建实验还验证了一个依赖风险：若只声明 `sentence-transformers`，当前 PyPI 解析可能拉取带 CUDA 依赖的 Torch。项目已显式固定 `torch==2.5.1+cpu` 和 CPU wheel index，最终镜像中 `torch.version.cuda` 为 `None`。

## 可靠性与回归

- 路由 Executor 有界并发测试连续 20 轮通过。
- 超时后的 permit 只在后台线程真实结束后释放，避免隐藏超卖。
- 慢 Embedding 期间 asyncio EventLoop 仍可调度。
- Agent 单元回归 181 条通过；宿主机缺 FastAPI 时跳过 3 条服务测试，Agent 容器内 8 条服务路由测试全部通过。
- 原 64 场景确定性 Agent 评测 64/64 通过；预测 grounding、citation、安全边界和 Skill 路由均为 100%。

## 运行与复现

```bash
scripts/prepare-routing-model.sh
make routing-unit
make routing-calibrate
make routing-evaluate
make routing-load-smoke
```

模型准备脚本是显式运维步骤；服务启动使用 `local_files_only`，Compose 同时设置 Hugging Face/Transformers 离线模式，并把模型缓存只读挂载到容器。

## 推广判定

当前候选已通过安全、已知意图改进、歧义回退、p95 小于 100 ms 和确定性 Agent 评测门槛。最终推广决定留到 Task 9 的完整验证和定向真实 LLM 回归后执行。无论是否推广，`rule` 模式始终保留为零 Embedding 依赖的兼容路径。
