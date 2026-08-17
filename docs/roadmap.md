# treeSem 工程路线与当前状态

## 模块状态

| 模块 | 内容 | 状态 |
|---|---|---|
| M0 | 基线、架构契约、文档与检查点 | 已完成 |
| M1 | C++ Controller/Service/Model/Infrastructure 分层 | 进行中 |
| M2 | Serving Bundle、Python 黄金实现、反归一化 | 待执行 |
| M3 | C++ ONNX Runtime 默认模型主链 | 待执行 |
| M4 | C++ 业务服务、Session、MySQL、医生反馈 | 待执行 |
| M5 | Python Agent Core 与领域 Tool | 待执行 |
| M6 | Doctor/Patient/Admin 权限、安全和审计 | 待执行 |
| M7 | 医疗知识 RAG 与 MCP Server | 待执行 |
| M8 | 医疗领域 Skill 和渐进加载 | 待执行 |
| M9 | Trace、Evaluation、压测和 RabbitMQ 决策门 | 待执行 |
| M10 | Docker、Demo、项目文档和面试材料 | 待执行 |

## M0 Definition of Done

- [x] 当前 C++ 和 Python 测试全部通过。
- [x] 真实 PPH 模型端到端返回 200。
- [x] 非法请求返回 400。
- [x] Adapter 不可用返回 502。
- [x] Worker 过载返回 503。
- [x] Adapter 超时返回 504。
- [x] 日志不记录完整预测响应。
- [x] API、错误码、线程模型和能力归属有文档。
- [x] 新服务命名使用 `treeSem`，历史兼容包不改名。
- [x] 创建本地可回滚基线提交，不推送远程。

## 当前停止线

M0/M1 完成前不加入 ONNX、MySQL、Agent、MCP/RAG、Skill 或医疗权限，避免在不稳定的后端边界上并行扩展。M1 完成后，M2 可以在不修改 HTTP 与业务层的前提下扩展模型输入输出。
