# treeSem M3 API 契约

本契约描述 M3 已实现的预测链路。未知的医疗单位保持 `null`，不根据字段名猜测。

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
  "fallback_enabled":true
}
```

该接口是已初始化进程的存活检查，不执行真实推理，也不同步探测 Python Adapter。ONNX 或 Bundle 初始化失败时进程不会开始监听。

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
| 404 | `not_found` | 路由不存在 |
| 500 | `model_inference_failed` | 严格 ONNX 模式的本地推理异常 |
| 500 | `internal_error` | 未分类内部异常 |
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
| `TREESEM_MODEL_BACKEND` | `onnx_fallback` |
| `TREESEM_SERVING_BUNDLE_DIR` | 无；ONNX 模式必须设置 |
| Python Adapter host | `127.0.0.1` |
| Python Adapter port | `18081` |

所有端口必须在 `1..65535`，所有 timeout、Worker 数和队列容量必须为正整数。非法配置在启动阶段失败。

`TREESEM_MODEL_BACKEND` 支持：`remote` 只走 Python；`onnx` 严格本地推理；`onnx_fallback` 在 ONNX 运行异常时调用一次 Python；`shadow` 返回 ONNX 并同步比较 Python。非法用户输入与损坏 Bundle 不触发 fallback。
