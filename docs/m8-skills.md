# M8：可信 Skill 与渐进加载

## Skill 解决什么

Tool 是原子能力，MCP 是外部能力的标准接入，Skill 是完成一类任务的可复用方法。Skill 不新增权限，也不能直接执行代码；它只声明需要哪些既有 Tool、知识 scope 和角色化回答步骤。

首批三个 Skill：

- `explain_prediction`：读取预测与解释，必要时检索模型术语，区分模型事实和通用说明。
- `compare_prediction_history`：读取历史并调用 C++ 确定性比较，避免 LLM 自算概率差和路径变化。
- `pph_evidence_education`：只用经引用的患者/医生知识回答一般 PPH 教育问题。

## 可信包与 Loader

每个本地包只包含 `skill.yaml`、Patient/Doctor instructions 和 `examples.json`。Loader 在 Agent 启动时检查受信根目录、符号链接、目录穿越、UTF-8、大小、严格字段、SemVer、角色、知识 scope 和已注册 Tool。所有文件 SHA-256 共同生成确定性 `skill_catalog_version`。

manifest 不允许声明 Python/Shell/SQL、URL、环境变量、依赖或网络入口。Skill 不能注册 Tool，也不能扩展 C++/MCP Capability。更新需要重启 Agent，避免运行中 Prompt 与版本漂移。

## 渐进加载

初始 Prompt 只包含角色可见的 Skill ID、用途、意图示例和 Tool 摘要。LLM 需要复杂流程时调用内部控制 Tool `activate_skill`；一个 Run 最多成功激活一个 Skill。激活后才读取该角色的完整 instructions，并把可见 Tool 收窄到 manifest 的 `required_tools`。

Tool 白名单在“发给 LLM 的 schema”和“执行入口”两处都校验，因此即使模型伪造隐藏 Tool Call 也会被拒绝。Skill instructions 是可信 System Instruction，但只能收窄基础医疗边界，不能覆盖 grounding、RBAC、Capability 或超时预算。

## 可追溯性

最终 Agent 响应和 Run 快照保存 Skill ID、版本和 Catalog 版本。幂等重放直接返回原版本；Skill 文件后来更新不会重跑旧请求。未激活 Skill 的简单问答保持 M5 行为兼容。

## 测试入口

测试覆盖三个包和每包至少 8 个固定示例、角色可见性、恶意路径/符号链接、未知 Tool、渐进 Prompt 注入、一个 Run 一个 Skill、执行端 Tool 收窄、grounding 和版本持久化。真实 LLM 的表达质量继续留到 M9 评测，CI 使用确定性 Fake LLM。
