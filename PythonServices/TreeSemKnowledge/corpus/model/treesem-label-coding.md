# treeSem PPH 标签编码与推理机制 / Label coding and inference

This project-owned reference describes the numerical serving contract, not a clinical diagnosis dictionary. It applies to the PPH serving implementation audited on 2026-09-14.

## 标签 0 与 1 / Negative and positive model classes

标签 label=0 是模型负类 negative；标签 label=1 是模型正类 positive。PPH 训练数据加载和 Bundle 导出均把源标签 -1 映射为 0，源标签 1 保持为 1。Bundle v2 的 label_schema 记录 negative=0、positive=1、source_negative_value=-1 和 threshold=0.5。这里说明的是项目数值编码；尚无权威临床数据字典证明该数据集的具体结局定义、观察窗口或判定标准。

Label 0 means the negative model class; label 1 means the positive model class. The source label -1 is mapped to 0, and source label 1 remains 1. These class names must not be presented as proof that a patient has no disease or as a confirmed diagnosis. 标签 0 不保证未患病，标签 1 不代表确诊；预测是辅助信息，不代替医生判断。

## 阳性概率与置信度 / Probability and confidence

positive_probability 是神经网络分类头 Softmax 输出中索引 1 的概率。confidence 是被选中类别的概率；label 取两个类别概率的 argmax。完全相等时实现选择索引 0，因此不能把规则无条件描述为 positive_probability >= 0.5 就输出 1。0.5 是现有二分类基线边界，不是经临床验证的治疗或诊断阈值。

positive_probability is the neural classifier's probability for class 1. confidence is the probability of the predicted class, not evidence reliability or a clinically calibrated guarantee. The predicted label uses argmax; an exact tie selects class 0.

## 神经网络与解释树 / Neural classifier and explanation tree

模型 label、positive_probability 和 confidence 来自神经网络分类头。tree_probability、tree_leaf_id 和 decision_path 来自独立的原生决策树计算。重要特征和决策路径可用于解释模型关联，但不是因果关系或个体治疗依据。不能断言“树叶变化导致神经网络阳性概率变化”；两者是不同输出链路。概率变化也不能单独证明病情改善或进展。

The neural classifier produces label, positive_probability and confidence. The native decision tree separately produces tree_probability, tree_leaf_id and decision_path. A change in tree leaf is not a demonstrated cause of a change in the neural probability. Explanation associations do not establish causation or clinical progression.

## 证据范围与溯源 / Provenance and limitations

编码事实来自历史 PPH 数据加载器、当前 Bundle 导出器的 label_schema，以及 Bundle predictor 的类别选择和树求值逻辑。本资料未补充临床结局定义、单位、类别语义、用药或手术建议。若问题要求这些缺失含义，应说明没有相应的可靠数据字典，而不是把项目编码当作医学定义。

Review anchors: historical PPH loader; TreeSemModelAdapter bundle exporter and Bundle v2 validation; Bundle predictor's neural argmax and separate native tree evaluation. The audited implementation is recorded in the companion regression report. This reference supplies encoding and serving-mechanism evidence only; missing clinical endpoint definitions remain unavailable.
