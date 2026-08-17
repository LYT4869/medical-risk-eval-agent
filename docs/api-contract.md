# treeSem M4 API 契约

本契约描述 M4 已实现的模型与持久化业务链路。未知的医疗单位保持 `null`，不根据字段名猜测。

## C++ Backend

### `GET /health`

成功响应：

```json
{
  "status":"ok",
  "service":"treeSem-backend",
  "model_version":"pph-seed42-1a299a474ce5",
  "configured_backend":"onnx_fallback",
  "primary_backend":"onnx",
  "fallback_enabled":true,
  "storage_backend":"mysql",
  "database_pool_size":8,
  "session_ttl_seconds":3600
}
```

该接口是已初始化进程的存活检查，不执行真实推理，也不同步探测 Python Adapter。ONNX 或 Bundle 初始化失败时进程不会开始监听。

### `POST /api/v1/predictions`

公开预测入口。

### `POST /internal/v1/predictions`

为后续 Agent Tool Adapter 保留的内部预测入口。M0/M1 中与公开入口使用同一业务服务；路径名称本身不构成安全隔离。

两个预测入口使用相同模型请求和响应契约。Public 使用 `treeSemSession` Cookie；Internal 必须提供 `X-TreeSem-Session-Id`。成功响应额外包含 `prediction_id`、`session_id` 和 UTC `created_at`。

#### 请求形式一：测试集样本下标

```json
{"sample_index":0}
```

- `sample_index` 必须是非负 JSON integer。
- 上界由持有数据集的 Model Adapter 校验。

#### 请求形式二：已预处理特征

```json
{"preprocessed_features":[49个有限数值]}
```

- 数组必须恰好包含 49 个有限数值。
- 该输入是内部兼容接口；调用方必须使用与当前 PPH 产物完全一致的特征顺序和标准化方式。

#### 请求形式三：原始命名临床特征

```json
{"raw_features":{"Gestational_Age":39,"...其余字段...":0,"Intrapartum_Bleeding":200}}
```

- `raw_features` 必须是 object，并严格包含 Bundle 中定义的 49 个大小写敏感字段。
- 不允许缺失字段、未知字段、布尔值、字符串、`null` 或非有限数值。
- 当前只做结构与数值校验；权威数据字典到位前不伪造医学范围或枚举规则。
- 字段顺序由 Bundle Schema 决定，标准化由具体模型服务完成。

请求必须是 JSON object，并且只能提供上述三个输入来源中的一个。重复 JSON key 会被拒绝。

#### 成功响应

响应保留以下顶层字段：

```json
{
  "model": "treeSem",
  "model_version": "pph-seed42-1a299a474ce5",
  "serving_backend": "onnx",
  "dataset": "pph",
  "input_source": "pph_test_split",
  "sample_index": 0,
  "prediction": {
    "label": 0,
    "positive_probability": 0.1,
    "confidence": 0.9,
    "cluster_id": 1,
    "tree_probability": 0.08,
    "tree_leaf_id": 8
  },
  "important_features": [],
  "decision_path": []
}
```

`important_features` 在原有字段之外包含 `display_name`、`original_value` 和 `unit`。分支路径同时返回 `feature_display_name`、`threshold_original`、`value_original` 和 `unit`。原有标准化字段保持不变；未知单位为 JSON `null`。

## 持久化业务 API

- `GET /api/v1/predictions/:prediction_id`
- `GET /api/v1/predictions/:prediction_id/explanation`
- `GET /api/v1/sessions/current/history?limit=20&cursor=...`
- `POST /api/v1/comparisons`
- `GET /internal/v1/predictions/:prediction_id`
- `GET /internal/v1/explanations/:prediction_id`
- `GET /internal/v1/sessions/:session_id/history?limit=20&cursor=...`
- `POST /internal/v1/comparisons`
- `POST /internal/v1/predictions/:prediction_id/feedback`
- `GET /internal/v1/predictions/:prediction_id/feedback`

History 的 `limit` 范围是 1～100，cursor 是 opaque base64url keyset cursor。比较请求包含 `prediction_id_a` 和 `prediction_id_b`，差值由 C++ 确定性计算，不重新推理。

内部反馈要求 Session Header 和 8～64 字符 `Idempotency-Key`。首次创建返回 201；相同 key 与 payload 重放返回 200；key 相同而 payload 不同返回 409。`reviewer_verified` 在 M4 永远为 false，且没有 public feedback 路由。

### `GET /ready`

该异步接口经 Database Scheduler 获取连接并执行 `SELECT 1`。成功返回 200；队列满、连接池超时或数据库不可用返回 503。`/health` 始终不查询数据库。

## Python Model Adapter

### `GET /health`

Bundle 模式返回服务、数据集、输入维度、reference 样本数、`model_version` 和 `serving_backend`。只有 Schema、checksum、维度、权重和树全部验证成功后才监听。

### `POST /v1/predict`

使用与 C++ 预测入口相同的三种请求形式，并返回相同的预测业务字段。该端口默认只监听 `127.0.0.1`。

## 错误契约

统一格式：

```json
{"error":"stable_error_code","message":"safe human-readable message"}
```

`message` 可以省略以兼容旧响应；不得包含本地路径、堆栈、患者原始数据或下游错误正文。

| HTTP | error | 含义 |
|---:|---|---|
| 400 | `invalid_json` | 请求体不是合法 JSON |
| 400 | `invalid_request` | JSON 类型、字段组合、维度或数值非法 |
| 400 | `invalid_session` | Session Header/Cookie 缺失或格式非法 |
| 404 | `not_found` | 路由不存在 |
| 404 | `resource_not_found` | Session/Prediction 不存在或跨 Session 查询 |
| 409 | `session_conflict` | Session 在推理期间过期或并发状态冲突 |
| 409 | `idempotency_conflict` | 幂等键被不同 payload 重用 |
| 500 | `model_inference_failed` | 严格 ONNX 模式的本地推理异常 |
| 500 | `internal_error` | 未分类内部异常 |
| 502 | `model_adapter_unavailable` | Adapter 无法连接或传输失败 |
| 502 | `invalid_model_adapter_response` | Adapter 响应为空、过大、非法或缺少必需字段 |
| 502 | `model_adapter_failed` | Adapter 返回 5xx |
| 503 | `prediction_overloaded` | 预测有界队列已满 |
| 503 | `database_overloaded` | 数据库有界队列已满 |
| 503 | `database_busy` | 连接池获取超时 |
| 503 | `database_unavailable` | MySQL 不可用 |
| 500 | `persistence_error` | 数据损坏或未分类持久化失败 |
| 503 | `service_stopping` | Worker Pool 已停止接单 |
| 504 | `model_adapter_timeout` | 连接或请求超过配置超时 |

## 默认配置

| 配置 | 默认值 |
|---|---|
| C++ 监听端口 | `8080`，也可使用第一个命令行参数覆盖 |
| `TREESEM_MODEL_ADAPTER_URL` | `http://127.0.0.1:18081/v1/predict` |
| `TREESEM_MODEL_CONNECT_TIMEOUT_MS` | `500` |
| `TREESEM_MODEL_TIMEOUT_MS` | `5000` |
| `TREESEM_INFERENCE_WORKERS` | `2` |
| `TREESEM_INFERENCE_QUEUE_CAPACITY` | `32` |
| `TREESEM_MODEL_BACKEND` | `onnx_fallback` |
| `TREESEM_SERVING_BUNDLE_DIR` | 无；ONNX 模式必须设置 |
| `TREESEM_STORAGE_BACKEND` | `mysql` |
| `TREESEM_DATABASE_WORKERS` | `4` |
| `TREESEM_DATABASE_QUEUE_CAPACITY` | `64` |
| `TREESEM_DB_POOL_SIZE` | `8` |
| `TREESEM_DB_ACQUIRE_TIMEOUT_MS` | `500` |
| `TREESEM_SESSION_TTL_SECONDS` | `3600` |
| `TREESEM_COOKIE_SECURE` | `false` |
| Python Adapter host | `127.0.0.1` |
| Python Adapter port | `18081` |

所有端口必须在 `1..65535`，所有 timeout、Worker 数和队列容量必须为正整数。非法配置在启动阶段失败。

`TREESEM_MODEL_BACKEND` 支持：`remote` 只走 Python；`onnx` 严格本地推理；`onnx_fallback` 在 ONNX 运行异常时调用一次 Python；`shadow` 返回 ONNX 并同步比较 Python。非法用户输入与损坏 Bundle 不触发 fallback。
