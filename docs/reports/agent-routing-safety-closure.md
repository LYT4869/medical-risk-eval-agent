# Agent 路由与安全策略收口报告

评测日期：2026-09-09

## 候选版本冻结

本轮先在 120 条 Calibration 上完成策略与阈值选择，再冻结代码和配置；240 条 Heldout 在冻结前未被读取或执行。以下信息用于标识即将接受一次性 Heldout 验收的候选版本：

- 候选 Git commit：`a93d91d7ead02b0303cf72c9f2a6cf7a304f7e41`。
- Task Registry SHA-256：`b80d66f8b079a24538929f22caab0e990a0d01073bde8c32eb59a591f69b3315`。
- 路由阈值 SHA-256：`bdbe77e6d074d92a5c3b97d41d3ed83af789af09b2641fb196a2c9fe5f8bf9a9`。
- Routing Quality Set SHA-256：`02aa8a89bb4265586034e995008aa7045a66c2ece3fcdf109567a18db370ca4b`。
- FP32 Artifact：`routing-7cf6f4740d6c69a7`。
- Artifact manifest SHA-256：`41800df906ad963c12c22212bc7a8f45383ada4c69f6706ac6769c624dba76b5`。
- ONNX model SHA-256：`4d4d1a462dd602bef691aa8920bae607ad73525a45e21e60d805ab5076f5ce62`。
- 意图向量 SHA-256：`fa4ab88c8ed8dab67f3f56d9e027166d36b07ca87b4d7ac9418b6a86f0bb3b81`。

Artifact 同时在 manifest 中绑定当前 Task Registry 与阈值校验和。生成目录属于本地忽略产物，不向 Git 提交 470 MB ONNX 文件。

## 设计收口

请求处理顺序固定为：

```text
概念型确定性安全策略
  → 高精度规则快路径
  → 有界 FP32 ONNX 语义路由
  → 低置信、歧义、组合、超时或过载时回到受限 Open Agent
```

Task Registry 是业务工作流静态元数据的唯一来源；运行时可用 Tool 仍取 Registry、RBAC/Capability、Skill 和阶段状态的交集。语义路由只能选择已注册业务范围，不能产生安全结论或授予权限。

安全层不依赖无限追加整句模板，而使用有限的“危险动作 + 目标 + 上下文”概念组合。它覆盖凭据泄露、权限绕过、事实/引用伪造、个体化用药与手术决定、确定性诊断、治疗保证和当前紧急症状；教育、解释和防御性问法由成对反例保护。

## Calibration 结果

Calibration 共 120 条，包含已知单意图、未知意图、跨工作流组合请求和安全请求。阈值只由该 split 选择：

| 指标 | Rule | Hybrid FP32 候选 |
|---|---:|---:|
| 已知单意图准确率 | 15.00% | 30.00% |
| 业务 Macro F1 | 23.67% | 41.20% |
| Unknown 召回率 | 95.00% | 95.00% |
| 组合请求回退率 | 95.00% | 95.00% |
| 安全准确率 | 100.00% | 100.00% |
| 非安全请求确定性覆盖率 | 17.00% | 28.00% |

最终阈值：最低相似度 `0.876893`、最低 margin `0.005699`、次意图相似度上限 `0.875226`。系统追求高精度路由而不是最大覆盖率，因此没有越过门槛的请求会交给受 Tool、步骤和 deadline 限制的 Open Agent。

## 部署一致性与冻结前回归

- SentenceTransformer 与 FP32 ONNX：150/150 最终路由一致。
- 最大 embedding 绝对误差：`1.70e-7`。
- 最小 embedding cosine：`0.9999999999996`。
- 最大 Top-1 相似度差：`1.08e-7`。
- 最大 margin 差：`1.09e-7`。
- 路由专项测试：92/92 通过。
- Guard、Workflow、Agent Loop 等相关测试：77/77 通过。
- 确定性 Agent 评测：64/64 通过；安全、prediction grounding、citation validity 均为 100%。

## Heldout 晋级结论

尚未执行。候选版本冻结后只运行一次 240 条 Heldout；无论结果如何，不使用 Heldout 继续调参或补规则。晋级要求为：安全 100%、无良性安全误拒、Unknown 与组合回退均不低于 95%，同时业务 Macro F1 与已知意图准确率有实质提升。
