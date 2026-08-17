# M1 C++ Backend 实现说明

## 交付结果

M1 将原先同时承担 JSON、HTTP、下游调用和结果透传的 `PythonModelProxy` 拆成四个可独立测试的层。旧类已经移除，不保留双入口。

| 层 | 主要类型 | 只负责什么 |
|---|---|---|
| API | `PredictionController`、`PredictionJsonCodec`、`HttpErrorMapper` | HTTP 边界、JSON 契约、响应与状态码 |
| Application | `PredictionService` | 单一预测用例的业务编排 |
| Model Abstraction | `IModelService`、`ModelInput`、`ModelResult`、`ModelException` | 与传输和运行时无关的模型能力契约 |
| Infrastructure | `RemoteTreeSemModelService`、`IModelAdapterClient`、`PythonModelClient`、`InferenceScheduler` | Adapter 协议、libcurl、线程调度和进程信号 |

配置由 `TreeSemServerConfig` 在启动时一次性读取和校验。`main.cpp` 是 composition root，只装配依赖、注册路由并控制服务生命周期。

## 一次预测的代码调用链

```text
HttpServer EventLoop
  -> PredictionController::handle
  -> PredictionJsonCodec::parseInput
  -> InferenceScheduler::schedule
  -> BoundedWorkerPool worker
  -> PredictionService::predict
  -> IModelService::predict
  -> RemoteTreeSemModelService::predict
  -> IModelAdapterClient::predict
  -> PythonModelClient::predict (libcurl)
  -> RemoteTreeSemModelService 校验并转换 ModelResult
  -> PredictionJsonCodec::serializeResult
  -> AsyncResponder / queueInLoop
  -> HttpServer 在连接所属 EventLoop 发送响应
```

这里最关键的边界是：EventLoop 只做轻量解析和调度，阻塞的 libcurl 调用只能发生在 Worker；Worker 不持有栈上的 `HttpResponse*`，只把 `ResponseWriter` 投回 I/O 线程。

## 为什么这样设计

1. `IModelService` 隔离模型运行时。M3 从 Python Adapter 切到 ONNX Runtime 时，只新增实现和调整 composition root，不修改 Controller 和业务层。
2. 同步领域接口配合异步边界。模型能力本身保持容易测试的同步接口，网络线程通过有界 Worker Pool 获得背压与隔离。
3. JSON 只存在于系统边缘。业务层使用 `std::variant` 和结构化结果，避免字符串字段在多层复制和漂移。
4. 下游响应不被盲目信任。字段缺失、错类型、超大整数、NaN/Inf、概率越界、特征索引越界和决策叶节点不一致都会映射为 502。
5. 错误正文统一构造。外部只看到稳定错误码和安全 message，内部路径、异常栈、libcurl 细节和患者数据不会回传。
6. 进程信号不直接执行复杂逻辑。SIGINT/SIGTERM 只向 non-blocking self-pipe 写一个字节，再由 EventLoop 正常回调停止服务；之后 Scheduler 排空已接任务。

## 测试覆盖

`ctest --test-dir build --output-on-failure` 当前包含 11 项：

- HTTP Parser、Router 和并发 one-shot responder 回归。
- BoundedWorkerPool 与 InferenceScheduler 的容量、停止和异常恢复。
- PredictionJsonCodec 两种输入、exactly-one、49 维、有限值、负数和超大整数。
- RemoteTreeSemModelService 的序列化、完整响应解析、状态映射和非法下游响应。
- PredictionController 使用 FakeModelService 验证 200/400/500/504，不访问网络。
- TreeSemServerConfig 默认值、覆盖值和 fail-fast 校验。
- Python Adapter 合约测试。
- 进程级 C++ Backend 测试：慢 Adapter 时健康检查不阻塞、并发请求不串线、队列满 503、客户端提前断开、敏感日志检查和优雅停机排空。

真实 PPH 验证仍使用 M0 的可信 PyTorch 产物。M1 只重构边界，不改变预测 URL、请求字段、响应业务字段和 Adapter 协议。

## 面试学习入口

这一阶段优先理解四个主题：

1. Reactor/EventLoop 为什么不能执行阻塞调用，以及 `queueInLoop` 如何保证 socket 线程归属。
2. 有界线程池如何实现背压，为什么队列满应该快速失败而不是无限堆积。
3. 依赖倒置：`PredictionService` 为什么依赖 `IModelService`，以及 ONNX 如何无侵入替换远程实现。
4. Controller、Service、Adapter/Repository 类似边界如何划分；哪些校验属于外部协议，哪些属于下游防御。

M2 学习时再把 Serving Bundle、Scaler 元数据、反归一化和模型版本管理接到这一架构上，不需要提前修改 M1 的分层。
