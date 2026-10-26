# TreeSem V2 M7-B｜P001 合成本地模型组件评测

本阶段不改 M1–M6 业务流程、C++ Gateway 或数据库。先隔离测 M3 Investigation Agent 的决策与 M5 Writer 的输出，记录 Prompt 口径、模型回复、校验结果和成本。**这里的测量仅代表六个固定决策和一个固定 Writer 合成任务，不是实际临床有效性、通用医学 NLI、完整 Agent Task Success 或端到端性能。**

## 设计与归因

- M3：六个 P001 Issue/Tool 决策样本，复用现有 OpenAICompatibleInvestigationAgent.SYSTEM、工作上下文字段和 checked_decision。输出分为 invalid_json、schema_mismatch、guard_rejected、golden_mismatch 和 golden_match。golden_match 只表示符合固定 Fake 基准轨迹，不等于真实通用推理能力。
- M5：用固定 10 条 P001 Fact 和六区 Section 结构调用现有 LocalParagraphWriter.SYSTEM；先做基础 JSON/Schema 检查，再调用原 review_document P001 Catalog。输出分类为 invalid_json、schema_mismatch、catalog_rejected、catalog_pass。**catalog_rejected 无法说明医学语义真假**：正确改写和错误事实都可能被固定长短文本目录拒绝，需要人工标注及后续真实 NLI 比较。semantic_correctness 固定为 not_evaluated。
- 记录返回全文（仅限 P001 合成输出）、Provider 声称的模型名、每次网络响应耗时，以及 API 若提供的 prompt_tokens、completion_tokens、total_tokens。缺少用量时记录 null；不估造 Token。文件原子保存、权限为 0600，默认保存在 Git 忽略的 artifacts 文件夹；可选 --redact-raw。
- LIVE 要求明确的本机 URL，域名只允许 localhost、127.0.0.1 或 ::1，必须带端口和 /v1/chat/completions 路径；没有自动外部云调用、凭证文件、患者病历加载、工具执行及报告发布。网络 Provider 错误单独归类。
- 该脚本复制现有适配器的 SYSTEM + user JSON 请求格式，以保持可比较的 Prompt 口径；但仍是**组件级独立调用**，不是 LangGraph 真正执行回合。

## 命令

CI 合成测试（不运行真实 LLM）：

~~~bash
export PYTHONPATH=PythonServices/TreeSemAgent
python -m unittest discover -s PythonServices/TreeSemAgent/v2_m7_eval/spec_tests -p 'test_*.py' -v
python -m v2_m7_eval.run_probe --mode fixture --out artifacts/m7b-fixture.json
~~~

自行启动并确认本地 OpenAI-compatible Qwen 服务后：

~~~bash
export PYTHONPATH=PythonServices/TreeSemAgent
python -m v2_m7_eval.run_probe --mode live --endpoint http://127.0.0.1:8000/v1/chat/completions --model YOUR_LOCAL_MODEL_ID --out artifacts/m7b-live.json
~~~

不用环境中的默认外部 API key，未确认本机 Serving 可访问时不要宣称已完成真实模型实验。实际模型精确版本、量化设置、硬件、样本数、JSON Raw、是否返回 usage 与服务端日志需写入实验台账。真实输出不得当作临床建议。

## 下阶段边界

待真实模型小样本测试有记录后再决定扩大样本、增加多轮 IssueResult 评测以及端到端 Writer Proposal 重放与真实 LLM 接入。需保留 M7-A 合成工程 Trace 与 M7-B 模型实验输出为两种不同证据等级，不将 fixture 成功计入 Agent 成功率。
