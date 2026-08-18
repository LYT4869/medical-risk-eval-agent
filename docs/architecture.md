# treeSem 服务平台架构

## 当前 M6 架构

当前可运行链路是：

```text
Client
  -> Muduo HttpServer / Router
  -> SecurityMiddleware（Access JWT / Capability / Origin）
  -> Prediction / Business / Chat / Auth / Doctor / Admin Controller
  -> Prediction / Database / Agent / Auth Scheduler（四个隔离的有界池）
  -> Prediction / Explanation / History / Comparison / Feedback / Agent / Auth Service
  -> SessionService + ITreeSemStore
  -> MySqlTreeSemStore（默认）/ InMemoryTreeSemStore（测试）
  -> PredictionService 调用 IModelService
  -> backend routing: onnx / onnx_fallback / shadow / remote
  -> OnnxTreeSemModelService（默认主链）
     -> ONNX Runtime Session（神经网络）
     -> FeaturePreprocessor + NativeDecisionTree（标准化与解释）
  -> RemoteTreeSemModelService（黄金参考与 fallback）
     -> PythonModelClient -> treeSem Python Bundle Adapter
  -> Prediction 与 Session 状态短事务提交
  -> Python Agent Core -> native domain tools -> C++ Internal API
  -> response queued back to the connection EventLoop
```

网络 EventLoop 只负责连接、HTTP 解析、轻量认证中间件、路由和响应发送。推理、Argon2、LLM/Tool 等待、连接池等待和 SQL 都在对应的有界 Worker Pool 中执行；Worker 通过一次性 `AsyncResponder` 把轻量响应构造任务投递回连接所属 EventLoop，不跨线程传递栈上的 `HttpResponse*`。

## M1 正式分层

```text
API Layer
  PredictionController / PredictionJsonCodec / HttpErrorMapper
                         |
Application Layer       v
  PredictionService
                         |
Model Abstraction       v
  IModelService / ModelInput / ModelResult / ModelException
                         |
Infrastructure          v
  RemoteTreeSemModelService
  IModelAdapterClient / PythonModelClient
  BlockingTaskScheduler / BoundedWorkerPool
  MySqlTreeSemStore / MySqlConnectionPool
```

依赖只能自上而下。`PredictionService` 不依赖 HTTP、JSON、libcurl 或 Python 名称；`main.cpp` 只作为 composition root 读取配置、构造对象、注册路由并启动服务。

M3 已利用 `IModelService` 无侵入加入本地 ONNX、fallback 和 shadow；Controller 与业务层没有因模型运行时变化而修改。

## Agent 与安全边界

```text
Client
  -> C++ Gateway / Auth / RBAC / Business Services / MySQL
  -> Python Agent Core（短生命周期 Run）
  -> scoped native domain tools
  -> C++ Internal API -> ONNX / MySQL
```

当前依赖约束：

- 模型与持久化业务层不依赖 Agent、MCP 或 Skill 概念；只有独立 Agent Application/Client 负责下游编排。
- Python Agent 通过 C++ Internal API 使用确定性业务能力，不直接访问模型实例或 MySQL。
- LLM 不能提供 session、actor 或 subject；C++ 签发的 Capability 绑定本次 Run 和允许的 Tool。
- `OnnxTreeSemModelService` 是默认模型主链；Python Adapter 保留为黄金参考和 fallback。
- RAG 只提供模型知识、可信临床参考和患者教育，不生成预测事实。
- Patient、Doctor、Admin 使用资源级授权；Admin 默认不能读取临床内容。
- RabbitMQ 仅在压测证明存在批量、长任务、状态查询或跨进程重试需求后加入。

## 请求线程时序

```text
EventLoop          Worker              ONNX / Python fallback
   |                  |                       |
   | parse + route    |                       |
   |----------------->| schedule              |
   | returns to poll  |                       |
   |                  |---- local Run/HTTP --->|
   |                  |<--- result / error ----|
   |<-----------------| queueInLoop(writer)   |
   | build + send     |                       |
```

不变量：

- EventLoop 不执行模型推理或等待下游 HTTP。
- 队列有界，满载时立即返回 503。
- 下游连接和总请求均有超时。
- 每个异步请求最多响应一次。
- 客户端提前断开时不再写连接。
- 预测响应体不写入服务日志。
- SIGINT/SIGTERM 通过 self-pipe 转为 EventLoop 事件；EventLoop 退出后，Scheduler 停止接单并排空已接任务。

## 架构决策

1. C++ Gateway 是统一外部入口，模型和 Agent 都是受控下游能力。
2. 同步模型接口仅由 Worker 调用，使业务层保持简单，同时保护 EventLoop。
3. 模型抽象只表示确定性预测模型；LLM Agent 不实现 `IModelService`。
4. 新服务、接口和文档统一使用 `treeSem`；历史训练包和可信产物中的 `trivae` 名称不改动。
5. Scaler 和树是 Bundle 中的跨语言事实；ONNX 只负责确定性神经网络子图。
6. Skill 使用可信本地目录和渐进加载，不建设通用插件市场。

## 能力归属

原 Kama-HTTPServer 提供 Muduo HTTP 框架、基础 Router、Middleware、Session、SSL 和五子棋示例。treeSem 项目新增或改造的能力包括：

- 修复 HTTP 请求、响应和路由行为并补测试。
- 异步路由与一次性跨线程响应。
- 有界 Worker Pool 和推理调度。
- treeSem C++ 服务入口和模型 Adapter 客户端。
- Python treeSem Model Adapter 与真实 PPH 模型接入。
- 确定性 Serving Bundle、命名临床输入和反归一化解释。
- C++ ONNX Runtime 主链、原生决策树、fallback、shadow 与全量一致性验证。
- 类型化匿名 Session、Repository、MySQL 连接池与事务持久化。
- 预测详情、解释快照、keyset 历史、确定性比较和内部未认证医生反馈。
- Python Agent Loop、结构化领域 Tool、grounding policy、Run/Chat 持久化和独立 Agent 舱壁。
- Argon2id、Access/Refresh Token、Refresh Family 重用检测和首个 Admin bootstrap。
- Patient/Doctor/Admin 资源级授权、Doctor-Patient assignment 和 verified feedback。
- scoped Capability JWT、Agent service credential、Origin/Cookie 安全与最小化审计。
- 超时、过载、下游故障映射和敏感响应日志治理。

简历中应表述为“基于 Muduo/Kama-HTTPServer 二次开发”，不表述为从零自研完整 HTTP 框架。
