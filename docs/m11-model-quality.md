# M11：模型质量复现、Bundle v2 与发布治理

## 当前结论

当前正式 seed42 Artifact 可以复现，不需要因为旧文档中的错误指标重训：

```text
Accuracy             0.9637340497
Positive F1          0.625
AUC                  0.9287898542
AUPRC                0.7393756854
Sensitivity          0.7627118644
Specificity          0.9720279720
Balanced Accuracy    0.8673699182
Python/C++ max delta 1.79e-7（现有正式 Bundle）
```

混淆矩阵为 `[[1390, 40], [14, 45]]`。`scripts/model_quality_audit.py` 同时核对原始 CSV、Artifact、Bundle checksum、测试集重建、PyTorch Bundle 黄金输出和可选 C++ ONNX 输出。本机审计结果为 `reproducible=true`，决策为 `keep_current_model`。

这证明的是数据/部署链可复现，不代表模型已经达到生产医疗验证要求。正类样本较少，因此不能只看 Accuracy；F1、AUPRC、Sensitivity、Specificity、Brier 与 ECE 都进入报告。

## Bundle v2

新 Exporter 默认生成 schema v2，Python/C++ Loader 同时兼容 v1/v2。v2 新增：

- Artifact、原始数据、Feature Schema 和 Preprocessing SHA。
- label 编码、0.5 兼容阈值、split 参数和 train/test index fingerprint。
- Exporter 版本、训练代码 commit（无法确认时为 null）和运行库版本。
- Artifact 指标快照与 Exporter 对全部测试样本复算的指标。
- 本地评测 Bundle 可选 `reference_labels.i8`；生产 Bundle 与 reference matrix 一起排除。

Exporter 会对 1489 个样本实际推理，并在 Accuracy、AUC、AUPRC、Positive F1、Balanced Accuracy、Sensitivity 或混淆矩阵不一致时拒绝导出。不能再出现“ONNX 数值一致但标签和测试集错位”的假通过。

本机已完成 v2 导出、ONNX 导出和 C++ parity：ONNX 相对 PyTorch 最大概率差 `1.19e-7`，cluster logit 差为 0，模型指标完全对齐。

## 五种子报告

`scripts/multiseed_report.py` 要求显式提供 19/21/42/60/99 五个同配置 Artifact，拒绝混用实验。它从每个权重重新推理而不是只抄旧 JSON，输出均值、样本标准差、固定随机种子的 bootstrap 95% CI、Brier、ECE，以及选中特征的 pairwise Jaccard 作为解释稳定性证据。

本机已对现存的 `pph_5seed_h128_l64_independent_20260709` 历史同配置实验生成报告。它用于稳健性证据，不用于从测试集挑选正式模型；当前正式 seed42 是另一发布候选，不能用这组历史实验偷换发布结论。

| 指标 | 五种子均值 | 样本标准差 |
|---|---:|---:|
| Accuracy | 0.9562 | 0.0038 |
| AUC | 0.9199 | 0.0196 |
| AUPRC | 0.7194 | 0.0302 |
| Positive F1 | 0.5920 | 0.0223 |
| Sensitivity | 0.8000 | 0.0279 |
| Specificity | 0.9627 | 0.0042 |
| Brier Score | 0.0456 | 0.0037 |
| ECE（10 bins） | 0.1308 | 0.0122 |

## 发布和回滚

`scripts/promote_bundle.py` 校验完整 Bundle 后原子记录 current/previous/history。切换仍通过 `TREESEM_SERVING_BUNDLE_DIR` 加服务重启，不实现热更新。发布前顺序固定为 Bundle 导出、PyTorch 指标、Python ORT、C++ ORT、shadow、全量业务/安全回归；分类、cluster、解释路径、Schema 或资源使用异常时回滚上一 Bundle。

## 面试重点

- 部署 parity、模型质量和医学有效性是三种不同问题。
- 为什么类别不平衡时 Accuracy 会误导，AUC 与 AUPRC 各说明什么。
- 为什么阈值只能在训练/验证集选择，不能用测试集挑 seed。
- 数据集 SHA、split fingerprint 和 scaler checksum 如何防止“同名不同数据”。
- 模型为什么选择重启切换而不是热更新，以及如何降低回滚复杂度。
