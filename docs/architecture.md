# treeSem 服务平台架构

## 当前 M1 架构

当前可运行链路是：

```text
Client
  -> Muduo HttpServer / Router
  -> PredictionController / PredictionJsonCodec
  -> InferenceScheduler
  -> BoundedWorkerPool
  -> PredictionService
  -> IModelService
  -> RemoteTreeSemModelService
  -> IModelAdapterClient / PythonModelClient (HTTP + timeout)
  -> treeSem Python Model Adapter
  -> trusted PyTorch artifact + sklearn tree
  -> response queued back to the connection EventLoop
```

网络 EventLoop 只负责连接、HTTP 解析、路由和响应发送。模型 HTTP 调用在有界 Worker Pool 中同步执行；Worker 通过一次性 `AsyncResponder` 把轻量响应构造任务投递回连接所属 EventLoop，不跨线程传递栈上的 `HttpResponse*`。

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
  InferenceScheduler / BoundedWorkerPool
```

依赖只能自上而下。`PredictionService` 不依赖 HTTP、JSON、libcurl 或 Python 名称；`main.cpp` 只作为 composition root 读取配置、构造对象、注册路由并启动服务。

M1 已按该结构实现。`IModelService` 是后续 M3 增加 `OnnxTreeSemModelService` 的替换点；Controller 和业务层不需要随模型运行时变化。

## 长期目标

```text
Client
  -> C++ Gateway / Business Services / Session / MySQL
  -> Python Agent Core
  -> native domain tools + medical knowledge MCP/RAG
```

长期依赖约束：

- C++ Backend 不依赖 Agent、MCP 或 Skill 概念。
- Python Agent 通过 C++ Internal API 使用确定性业务能力，不直接访问模型实例或 MySQL。
- 后续 `OnnxTreeSemModelService` 是默认模型主链；当前 Python Adapter 保留为黄金参考和 fallback。
- RAG 只提供模型知识、可信临床参考和患者教育，不生成预测事实。
- Doctor、Patient、Admin 的认证、授权和审计属于后续安全模块。
- RabbitMQ 仅在压测证明存在批量、长任务、状态查询或跨进程重试需求后加入。

## 请求线程时序

```text
EventLoop          Worker              Python Adapter
   |                  |                       |
   | parse + route    |                       |
   |----------------->| schedule              |
   | returns to poll  |                       |
   |                  |---- HTTP POST ------->|
   |                  |<--- JSON / error -----|
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
5. M2 负责 Serving Bundle、反归一化和原始临床字段；M1 不临时复制这部分逻辑。
6. Skill 使用可信本地目录和渐进加载，不建设通用插件市场。

## 能力归属

原 Kama-HTTPServer 提供 Muduo HTTP 框架、基础 Router、Middleware、Session、SSL 和五子棋示例。treeSem 项目新增或改造的能力包括：

- 修复 HTTP 请求、响应和路由行为并补测试。
- 异步路由与一次性跨线程响应。
- 有界 Worker Pool 和推理调度。
- treeSem C++ 服务入口和模型 Adapter 客户端。
- Python treeSem Model Adapter 与真实 PPH 模型接入。
- 超时、过载、下游故障映射和敏感响应日志治理。

简历中应表述为“基于 Muduo/Kama-HTTPServer 二次开发”，不表述为从零自研完整 HTTP 框架。
