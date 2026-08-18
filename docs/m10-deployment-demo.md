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

`scripts/demo_flow.py` 自动完成 Admin、Doctor、Patient、assignment、两次预测、解释、比较、RAG 引用、医生反馈、横向越权 404，以及 Knowledge/Adapter 故障隔离。脚本使用幂等键并为每轮建立隔离账户。

## 验证边界

本机已通过 `prepare-demo` Artifact 预检和 `docker compose config --quiet`。Docker daemon 当前返回 permission denied，因此镜像真实构建、容器 E2E 和 Grafana 抓取必须在有 Docker 权限的终端运行；项目文档不会把静态检查描述成容器验收。

Fast CI 运行无外部 Artifact 的 C++/Python 单元测试、60 条确定性 Agent 评测、Compose 配置、密钥/大文件检查。`verify-full` 用于具备 ONNX Bundle、知识索引、MySQL 和 Docker 的本地环境。

## 面试重点

- build-time、run-time 与敏感 Artifact 为什么分离。
- readiness、liveness 和 Compose 启动依赖的区别。
- 为什么只暴露反向代理端口，内部 Agent/MCP/数据库不直接暴露。
- 如何做到离线演示可重复，同时不把 Scripted LLM 冒充真实模型质量。
- 为什么完整模型集成门槛放在具备私有 Artifact 的本地 Gate，而公共 CI 只做可复现子集。
