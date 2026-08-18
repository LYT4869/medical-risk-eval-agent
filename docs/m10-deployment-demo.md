# M10：部署、演示与项目包装

## 部署拓扑

```text
Browser :3000
  -> nginx static demo / reverse proxy
  -> C++ backend
       -> MySQL
       -> C++ ONNX Runtime
       -> Python model fallback
       -> Python Agent -> Knowledge MCP

optional: Prometheus -> backend/agent/knowledge metrics -> Grafana
```

`docker-compose.yml` 定义 MySQL、Backend、Model Adapter、Agent、Knowledge 和静态 Demo Web；Prometheus/Grafana 位于可选 profile。只有 Web 端口默认暴露到宿主机，内部服务位于独立 Compose 网络。模型 Bundle、知识索引和 Hugging Face 缓存只读挂载，原始 CSV/PDF 和密钥不进入镜像。

五个应用镜像均使用非 root 用户。C++ Backend 使用多阶段构建，固定 Muduo、ONNX Runtime 版本和下载 SHA；Python 服务使用锁定 requirements。`scripts/prepare_demo.py` 在启动前验证 Bundle、ONNX、索引及 checksum，生成权限为 0600 的本地 `.env`，不会在线替换医学资料或模型。

## 使用方式

```bash
make prepare-demo
make demo
make demo-flow

# 可选 Prometheus/Grafana
make demo-observability
```

默认 `scripted_demo` 只允许 local 环境，用于无外网的可重复面试演示；`demo-real` 要求提供 OpenAI-compatible URL、模型和 Key。离线客户端仍实际调用 Tool、MCP、Skill 与权限链，不伪造 C++ 预测结果。

演示页支持注册/登录、样本预测、解释、历史、比较、Agent Chat、citation、模型版本、Serving Backend 和 Trace ID；Doctor 入口可查看已分配患者并提交 verified feedback。它刻意不做 49 字段录入、完整 Admin 工作台或复杂前端框架。

`scripts/demo_flow.py` 自动完成 Admin、Doctor、Patient、assignment、两次预测、解释、比较、RAG 引用、医生反馈、横向越权 404，以及 Knowledge/Adapter 故障隔离。脚本复用固定演示账户，并为每个有副作用的请求生成独立幂等键。

## 验证边界

2026-08-18 已在本机完成真实容器验收，而不只是 Compose 静态检查：

- 五个应用镜像和 MySQL 8.0.42 均成功构建/拉取，六个主服务全部通过 healthcheck。
- 完整离线演示通过，覆盖 ONNX 预测、MySQL 持久化、三个 Skill、MCP/RAG 引用、医生反馈和横向越权 404。
- 同时停止 Knowledge 和 Python Adapter 后，C++ ONNX 主链仍成功预测。
- MySQL 重启前后 Prediction/Agent Run/Chat Message 计数均保持 `19/6/12`，Backend `/ready` 自动恢复为 200。
- Prometheus 的 Backend、Agent、Knowledge 三个 Target 全部为 `up`，Grafana 11.5.2 health 为 `database=ok` 且 Dashboard 完成预置。
- Backend 接收 SIGTERM 后 1 秒内以退出码 0 停止，随后能够正常恢复健康。
- 本地 CTest `27/27` 通过；60 条确定性 Agent Evaluation 全部通过，引用、grounding、医疗边界和跨角色隔离硬指标均为 100%。

容器联调额外发现并修复了两个只在干净镜像中暴露的问题：旧版 MySQL Connector/C++ 读取 JSON 类型结果时需要显式 `CAST(... AS CHAR)`，并且相邻查询前应及时释放 ResultSet；私有只读 Artifact bind mount 则通过宿主 UID/GID 映射读取，不放宽患者派生数据的文件权限。

Fast CI 运行无外部 Artifact 的 C++/Python 单元测试、60 条确定性 Agent 评测、Compose 配置、密钥/大文件检查。`verify-full` 用于具备 ONNX Bundle、知识索引、MySQL 和 Docker 的本地环境。

## 面试重点

- build-time、run-time 与敏感 Artifact 为什么分离。
- readiness、liveness 和 Compose 启动依赖的区别。
- 为什么只暴露反向代理端口，内部 Agent/MCP/数据库不直接暴露。
- 如何做到离线演示可重复，同时不把 Scripted LLM 冒充真实模型质量。
- 为什么完整模型集成门槛放在具备私有 Artifact 的本地 Gate，而公共 CI 只做可复现子集。
