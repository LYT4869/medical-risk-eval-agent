# M2：treeSem Serving Bundle 与黄金参考实现

## 为什么增加 Bundle

历史入口把 PT、研究源码和原始 CSV 一起带到 Serving 进程，并在启动时重新构造数据划分和 Scaler。它适合复现实验，却不适合稳定部署。M2 把推理所需事实固化为带版本和 checksum 的只读 Bundle，使加载错误在启动阶段暴露。

## 产物边界

Bundle 仅包含 49 维特征 Schema、Scaler 参数、确定性神经网络权重、跨语言决策树和精选黄金输出。Decoder、logvar、随机采样、Dropout、DeepSHAP buffer 和训练状态不进入 Serving。

模型版本使用 `pph-seed42-<artifact SHA256 前12位>`。业务文件采用确定性 JSON/NPZ 编码；相同 artifact 与 CSV 重复导出可逐文件验证一致。

`reference_inputs.f32` 是可选的本地测试资产。即使去掉直接身份字段，它仍是患者派生矩阵，所以默认不生成、不提交，也不能随生产包发布。

## 推理调用链

```text
命名原始字段 / 标准化数组 / reference index
  -> FeaturePreprocessor
  -> fc1 + ReLU -> fc21(mu)
  -> classifier + Softmax / cluster head
  -> 原生 TreeEvaluator
  -> 预测、重要特征、决策路径与反归一化展示
```

神经网络概率和 cluster 由 PyTorch 黄金引擎给出；树不再反序列化 sklearn 对象，而是执行 `tree.json`。这一边界让 M3 可以只替换神经网络执行器为 ONNX Runtime，解释逻辑保持同一份语义。

## 校验与安全

- Loader 校验 Schema 版本、文件大小、SHA256、49 维顺序、有限 mean/scale、权重键/shape/dtype 和树拓扑。
- 原始输入必须严格包含 Schema 中 49 个字段；无权威数据字典时不猜医学单位和取值范围。
- 日志不记录原始特征、完整预测结果或 Bundle 绝对路径。
- 历史训练包与产物中的 `trivae` 名称不修改；新服务、文档和 API 统一使用 `treeSem`。

## 面试可讲重点

这是“训练产物如何工程化上线”的完整切面：可复现版本、数据预处理固化、fail-fast 完整性验证、训练/Serving 解耦、黄金实现，以及向 C++ 跨语言迁移前的稳定契约。M3 的一致性验证不是拿旧训练脚本临时跑，而是以这个黄金实现为基准。
