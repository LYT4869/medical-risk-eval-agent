# treeSem M0/M1 API 契约

本契约只描述 M0/M1 已实现的预测链路。原始临床字段、原始单位和模型 Serving Bundle 在 M2 增量加入。

## C++ Backend

### `GET /health`

成功响应：

```json
{"status":"ok","service":"treeSem-backend"}
```

该接口是进程存活检查，不同步探测 Python Adapter。

### `POST /api/v1/predictions`

公开预测入口。

### `POST /internal/v1/predictions`

为后续 Agent Tool Adapter 保留的内部预测入口。M0/M1 中与公开入口使用同一业务服务；路径名称本身不构成安全隔离。

两个预测入口使用相同请求和响应契约。

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

请求必须是 JSON object，并且只能提供上述两个输入来源中的一个。可以携带未来兼容字段，但不能同时提供两个输入来源。

#### 成功响应

响应保留以下顶层字段：

```json
{
  "model": "treeSem",
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

当前 `important_features` 和 `decision_path` 使用标准化值。M2 会以新增字段的方式提供原始值、阈值、特征单位和显示文本。

## Python Model Adapter

### `GET /health`

返回服务、数据集、输入维度、测试样本数、设备和可信产物文件名。只有模型和数据加载成功后服务才开始监听。

### `POST /v1/predict`

使用与 C++ 预测入口相同的两种请求形式，并返回相同的预测业务字段。该端口默认只监听 `127.0.0.1`。

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
| 404 | `not_found` | 路由不存在 |
| 500 | `inference_failed` / `internal_error` | 未分类内部异常 |
| 502 | `model_adapter_unavailable` | Adapter 无法连接或传输失败 |
| 502 | `invalid_model_adapter_response` | Adapter 响应为空、过大、非法或缺少必需字段 |
| 502 | `model_adapter_failed` | Adapter 返回 5xx |
| 503 | `server_overloaded` | 有界队列已满 |
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
| Python Adapter host | `127.0.0.1` |
| Python Adapter port | `18081` |

所有端口必须在 `1..65535`，所有 timeout、Worker 数和队列容量必须为正整数。非法配置在启动阶段失败。
