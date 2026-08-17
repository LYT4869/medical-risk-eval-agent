# M3：C++ ONNX Runtime 主链

## 运行边界

ONNX 只承载 `fc1 -> mu -> classifier/cluster` 的确定性神经网络。Scaler 和 sklearn 树不塞入计算图：它们在 Bundle 中是可审计的跨语言数据，由 C++ `FeaturePreprocessor` 和 `NativeDecisionTree` 执行。这样既能保持解释路径完全一致，也避免把 Python/sklearn 对象带进 C++。

`OnnxTreeSemModelService` 在启动时加载一次 Bundle 和 Session。每次预测只创建请求自己的 vector/Tensor；Session 和只读树由 Worker 共享。ORT intra-op/inter-op 均为 1，避免“Worker Pool × ORT 线程池”导致 CPU 过度订阅。

## 后端策略

- `onnx`：严格本地主链，运行错误返回 `model_inference_failed`。
- `onnx_fallback`：默认。仅 ONNX Run、输出 shape 或非有限输出异常时调用一次 Python。
- `remote`：调试和 remote-only 构建兼容。
- `shadow`：先得到 ONNX 结果，再跑 Python；对外始终返回 ONNX，差异日志只含版本、差异类型和最大概率误差。

Bundle/checksum 错误属于启动错误，非法请求属于 400，两者都不 fallback。健康检查不做推理和远程探测；能监听即说明主 Bundle 与 ONNX Session 已完成初始化。

## 一致性结果

PPH seed42 的 1489 个 reference 样本比较了 PyTorch Bundle Reference、Python ORT、C++ ORT 和 Python Adapter：

- Python ORT 相对 PyTorch最大概率误差：`1.19e-7`。
- C++ ORT 相对 Python黄金实现最大概率/置信度误差：`1.79e-7`。
- label、cluster ID、tree leaf、decision path、重要特征 index/order：100% 一致。
- Accuracy `0.9019476`、F1 `0.0394737`、AUC `0.4324580` 在 Python/C++ 间完全一致。

这里的指标仅证明迁移没有改变当前模型语义，不表示医疗效果达标。模型质量评审和 Serving 工程正确性是两个独立问题。

## 面试可讲重点

这部分可以围绕“训练模型如何迁到 C++ 服务”展开：为什么只导确定性子图、如何固定预处理、为什么用黄金实现做 parity、如何处理 float32 数值容差、Session 生命周期与线程过度订阅、fail-fast、fallback 的触发边界，以及 shadow 灰度为什么不能记录患者输入。
