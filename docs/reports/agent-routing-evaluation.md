# Agent 混合路由评测与 ONNX 部署报告

评测日期：2026-09-08

## 结论

语义路由的生产实现已从 PyTorch/Sentence Transformers 迁移到本地 ONNX Runtime。FP32 ONNX 在冻结的 150 条用例上与 PyTorch 黄金实现达到 **150/150 最终路由一致**，因此作为后续质量评测的语义后端；动态 INT8 虽然更快、更小，但产生 6 条路由变化并让 Unknown 召回率从 100% 降到 95%，所以不进入候选主链。

系统默认仍为 `rule`，`hybrid_optional` 仍需显式开启。原因是此次工作证明了“如何低成本部署语义路由”，没有改变上一轮 held-out 质量结论：语义回退只把已知意图准确率从 56.67% 提升到 60.00%。是否推广为默认路由，要等独立的 300～400 条 Routing Quality Set 验证；该语料尚未生成。

最终结构保持不变：确定性安全策略优先，高精度规则走快路径，只有规则未命中时才进入有界 ONNX 语义执行器；歧义、组合意图、超时、过载或可选后端故障统一降级为受 Tool 权限约束的 Open Agent。语义路由只选择工作流，不能授予权限或扩大 Tool 集合。

## 可复现配置

- 模型：`intfloat/multilingual-e5-small`。
- Revision：`614241f622f53c4eeff9890bdc4f31cfecc418b3`。
- 冻结路由集：150 条合成、无患者数据用例；语料 SHA-256 为 `21f9b16c61b34c74f9b4365e1d545911e48e84eff152c3eb2f2af0717b4e849b`。
- 黄金后端：CPU PyTorch 2.5.1、Sentence Transformers 5.7.0。
- 部署后端：ONNX Runtime 1.20.1、Tokenizers 0.22.2、NumPy 2.0.1。
- ONNX 契约：opset 17、512 tokens、右侧 padding/truncation、`query: ` / `passage: `、attention-mask mean pooling、L2 normalize。
- Executor：1 Worker、队列容量 8、admission timeout 5 ms、route timeout 150 ms。
- 稳定化校准阈值：最低相似度 `0.880862`、最低 margin `0.002539`、次意图相似度上限 `0.918678`。

阈值仍只由 calibration split 选择。阈值从观测浮点边界调整到六位小数安全网格后，PyTorch 在原 150 条上的最终路由 **0 条变化**，避免正常的 PyTorch/ONNX 数值漂移跨过边界。Artifact 同时绑定 Task Registry、阈值文件、模型、Tokenizer、意图向量和黄金路由的 SHA-256；任一错配都会在启动阶段失败。

## 路由质量与后端一致性

### 原 held-out 质量结论

| 指标 | 高精度规则 | 规则 + E5 语义回退 |
|---|---:|---:|
| 已知单意图准确率 | 56.67% | 60.00% |
| 业务 Macro F1 | 69.71% | 72.69% |
| Unknown 召回率 | 100% | 100% |
| 组合意图回退率 | 100% | 100% |
| 安全用例准确率 | 100% | 100% |

这组结果用于回答“语义回退是否提高路由覆盖”；下面的 150 条全量 parity 用于回答“替换部署后端是否改变原行为”，两者不能混为同一个指标。

### PyTorch 与 ONNX parity

| 指标 | FP32 ONNX | INT8 ONNX |
|---|---:|---:|
| 最终路由匹配 | 150/150 | 144/150 |
| 最大 embedding 绝对误差 | `1.70e-7` | `0.02688` |
| 最小 embedding cosine | `0.9999999999996` | `0.98209` |
| 最大 Top-1 相似度差 | `1.08e-7` | `0.01268` |
| 最大 margin 差 | `1.09e-7` | `0.01447` |
| Unknown 召回率 | 100% | 95% |
| 组合意图回退率 | 100% | 100% |
| 安全准确率 | 100% | 100% |
| 晋级结果 | 通过 | 拒绝 |

INT8 的 6 条变化包括已知意图误路由和 1 条 Unknown 被误判为知识查询。项目没有通过放宽 Unknown 或安全门槛来保留 INT8；回滚到 FP32 是预先约定的发布规则。

## 资源与延迟证据

测量机器：双路 Intel Xeon Gold 6148、80 个逻辑 CPU、Linux x86_64。三组均使用同一组六条查询，100 次预热、1000 次单请求测量。PyTorch 基线在同一机器上重新运行；镜像与 Artifact 总体积是逻辑部署体积，不等同于 Docker 共享层后的物理增量。

| 项目 | PyTorch 语义后端 | FP32 ONNX | INT8 ONNX（未采用） |
|---|---:|---:|---:|
| 运行镜像 | 1,305,265,307 B | 382,582,166 B | 382,582,166 B |
| 模型/Artifact | 493,292,828 B | 487,493,305 B | 424,098,031 B |
| 逻辑总部署体积 | 1,798,558,135 B | 870,075,471 B | 806,680,197 B |
| 冷初始化 | 6,349.79 ms | 3,053.96 ms | 2,767.11 ms |
| 首次路由 | 20.27 ms | 12.38 ms | 8.41 ms |
| warmed p50 | 19.23 ms | 11.24 ms | 7.22 ms |
| warmed p95 | 21.25 ms | 11.89 ms | 8.16 ms |
| warmed p99 | 25.50 ms | 12.36 ms | 9.47 ms |
| 最大 RSS | 790,404 KiB | 1,224,084 KiB | 1,151,192 KiB |

相对 PyTorch，FP32 ONNX 将逻辑部署体积降低 **51.62%**、冷初始化降低 **51.90%**、warmed p95 降低 **44.05%**；但本次 ONNX Runtime 的最大 RSS 反而增加 **54.87%**。因此准确说法是“显著降低镜像/部署存储和路由延迟”，不能宣称所有资源维度都下降。INT8 的体积和延迟更优，但质量门槛失败，不能用资源收益覆盖错误路由风险。

## 生产包装与故障边界

- Agent 运行镜像只安装 ONNX Runtime、Tokenizers 和 NumPy，不包含 Torch、Sentence Transformers 或 Transformers。
- 重型导出器使用独立 Docker 镜像，只在显式运维命令中运行；Agent 启动过程不下载模型。
- ONNX、Tokenizer、意图向量和黄金路由组成不可变 Artifact，Compose 以只读 Volume 挂载。
- 目录权限固定为容器非 root 用户可读；导出器曾暴露 `0700` 目录导致容器无法加载，现已用发布权限测试覆盖。
- `rule` 模式不读取空 Artifact；`hybrid_optional` 初始化失败时降级；`hybrid_required` 初始化失败时拒绝启动。
- 语义计算继续运行在固定大小线程池中，带有有界 admission 和超时，不阻塞 asyncio EventLoop。

## Agent 回归与真实模型边界

原 64 场景确定性 Agent 评测为 64/64，预测 grounding、citation、安全边界和 Skill 路由均为 100%。上一轮真实 `qwen3.7-plus-2026-05-26` 定向评测两轮均为 6/8，失败集中在模糊指代和复杂组合规划。FP32 ONNX 与原 PyTorch 在冻结路由集上的最终路由完全一致，因此此次没有重复付费 LLM 评测；后端迁移不能解决 LLM 编排问题，也不应把一次等价部署改造包装成 Agent 质量提升。

## 运行与复现

```bash
# 显式导出，不在服务启动时执行
make routing-export-fp32
make routing-export-int8

# 指定生成目录与后端
make routing-parity \
  ROUTING_ARTIFACT_DIR=artifacts/agent-routing/<version> \
  ROUTING_BACKEND=onnx_fp32
make routing-benchmark \
  ROUTING_ARTIFACT_DIR=artifacts/agent-routing/<version> \
  ROUTING_BACKEND=onnx_fp32

make routing-unit
make routing-load-smoke
```

生成的 ONNX/F32 Artifact 和原始评测 JSON 均位于 Git 忽略目录；仓库只保存导出、校验、评测代码和可复现结论。

## 推广判定与下一检查点

- FP32 后端一致性：通过，作为语义候选后端。
- INT8 后端一致性：失败，拒绝晋级。
- 安全、Unknown 与组合意图硬门槛：FP32 全部通过。
- 运行时依赖瘦身：完成。
- 默认模式：仍为 `rule`。
- 下一门槛：独立生成并审核 300～400 条 Routing Quality Set，再比较 `rule` 与 `hybrid_optional`。

本阶段到这里停止，不自行调用 LLM 生成新语料。新语料应与当前 150 条校准/parity 集隔离，避免把针对已知失败的调试数据当成独立泛化证据。
