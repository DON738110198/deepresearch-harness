# Deep Research 项目调研与方向调整

调研时间：2026-08-13

## 1. 这次先定位的问题

当前项目把大量精力放在双盲逐 Claim 人工审核上，却还没有提供真正可用的在线搜索、网页读取和交互式研究体验。结果是：研究者需要手工标注 40 份候选报告，但用户仍不能输入一个现实问题并得到可追踪的网页研究报告。

这不是“用户不会使用工作台”，而是产品和实验顺序设计错误：**评测负担先于可用价值，并且人工成本超过了当前结论的价值。**

因此，B1/B2 v3 人工盲审暂停。已有工作台保留为可选审计工具，不再作为项目推进门槛，也不能产生“已完成人工验证”的表述。

## 2. 主流项目实际上怎样工作

| 项目 | 用户工作流 | 核心执行方式 | 常用评测方式 | 对本项目的启示 |
| --- | --- | --- | --- | --- |
| OpenAI Deep Research | 用户描述目标和约束，选择来源，审核/修改研究计划；运行中可查看进度并中断，最终得到带引用、来源列表和活动历史的报告 | 多步浏览与推理，具体内部 harness 未完全公开 | 标准 benchmark、安全评测、合成与人工 golden examples；不是让普通用户逐 Claim 标注每次运行 | 人应控制目标、来源和计划，而不是承担批量标注 |
| GPT Researcher | 输入查询，实时查看进度，得到可导出的长报告 | planner 生成子问题，execution/crawler 收集并总结来源，publisher 聚合报告 | 官方 README 重点是可用产品与来源追踪，没有把人工盲审设为日常使用流程 | 最小产品必须先完成真实搜索到报告的闭环 |
| LangChain Open Deep Research | 输入问题并配置模型、搜索工具或 MCP；可在 Studio/OAP 中运行 | Scope -> Research -> Write；当前实现允许研究 Agent 迭代搜索，旧实现包含 workflow 和 supervisor-researcher 两种方案 | DeepResearch Bench：对 100 个专家任务运行，使用基于专家报告的 LLM Judge 与 citation 评测 | 批量比较应自动化；多 Agent 是可选实现，不是项目成立条件 |
| Hugging Face open DeepResearch | Agent 自主调用网页、文本和代码工具回答复杂问题 | 简单 CodeAgent + 浏览/文本处理工具，重点是工具质量和可执行推理 | GAIA 的标准答案自动计分 | 有明确答案的任务优先用确定性指标，不需要人工审长报告 |
| DeerFlow 2 | 对话、计划模式、进度、文件与最终产物；必要时请求澄清 | lead agent + skills/tools + sandbox/memory；复杂任务才委派并行 subagents | 2026 年评测平台仍在建设，第一阶段强调真实 Gateway 执行、事件/产物和 deterministic oracle | 借鉴运行时边界、可观察性和预算保护，不照搬其整套重型架构 |
| Tongyi DeepResearch | 运行 ReAct 或更重的 IterResearch/Heavy 推理模式 | 模型训练与 test-time scaling 并重 | BrowseComp、HLE、WebWalkerQA、FRAMES、SimpleQA 等带参考答案 benchmark 自动评测 | 它的训练路线与本项目冻结参数的定位不同，只参考推理和 benchmark 接口 |

来源：

- OpenAI Deep Research 使用流程：https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt
- GPT Researcher 架构：https://github.com/assafelovic/gpt-researcher
- LangChain Open Deep Research：https://github.com/langchain-ai/open_deep_research
- LangChain 从零实现教程：https://github.com/langchain-ai/deep_research_from_scratch
- Hugging Face open DeepResearch：https://github.com/huggingface/blog/blob/main/open-deep-research.md
- DeerFlow 2：https://github.com/bytedance/deer-flow
- DeerFlow Evaluation RFC：https://github.com/bytedance/deer-flow/issues/4083
- Tongyi DeepResearch：https://github.com/Alibaba-NLP/DeepResearch

## 3. 市面上的共同主干

不同项目的名字很多，但最小主干基本一致：

```text
目标与约束
  -> 必要时澄清
  -> 可查看/修改的研究计划
  -> 迭代 Search + Fetch + Read
  -> 压缩并保留来源的研究笔记
  -> 按信息缺口继续搜索或停止
  -> 带行内引用的结构化报告
  -> 可查看来源、轨迹、Token、费用和延迟
```

真正有价值的技术问题主要是：

1. 什么时候继续搜索，什么时候停止。
2. 如何避免重复搜索、上下文膨胀和无关来源。
3. 如何让报告中的结论能够回溯到网页证据。
4. 如何在固定 Token 或费用预算下分配搜索深度。
5. 失败以后能否通过 trace 判断是搜索、读取、规划还是写作出了问题。

“是否用了多 Agent”不是第一问题。只有当坏例证明多个相互独立的研究分支串行执行太慢，或者单一上下文发生严重干扰时，才应该加入并行 subagents。

## 4. 主流项目怎样避免人工逐条审核

### 4.1 有标准答案的任务

GAIA、BrowseComp、SimpleQA 等任务可以使用 exact match、规范化答案匹配或官方 evaluator。人工主要参与 benchmark 构建，而不是每次实验运行。

### 4.2 长报告任务

DeepResearch Bench 使用两类自动评测：

- RACE：依据专家参考报告，为每个任务动态生成完整性、分析深度、指令遵循和可读性标准，再由 LLM Judge 评分。
- FACT：提取 Claim-URL 对，抓取引用网页，判断网页是否真正支持 Claim。

DeepResearch Bench II 更进一步，把专家报告分解成 9,430 条细粒度、可验证 rubric。Judge 必须为每条标准输出分数、理由和报告内证据，再聚合成任务与维度分数。

来源：

- DeepResearch Bench：https://github.com/Ayanami0730/deep_research_bench
- DeepResearch Bench II：https://github.com/imlrz/DeepResearch-Bench-II

这些 benchmark 仍然存在 Judge 偏差，所以高可信结论需要校准或抽查；但它们没有要求单个开发者手工标注所有候选。正确的人机分工是：

```text
确定性检查全量运行
  + 固定 rubric 的独立 LLM Judge 全量运行
  + 只抽查 Judge 分歧、证据抓取失败和关键坏例
```

## 5. 对当前项目的明确判断

### 做对的部分

- 冻结模型、固定预算、记录 Token/费用/延迟的比较边界是必要的。
- Claim-Evidence Ledger 和 Evidence Debt 适合作为内部审计与停止策略的基础。
- B2 在 Token-matched 下通过自动 gate、在 cost-matched 下失败的负结果应保留。
- Pydantic state、trace 和 provider 抽象适合作为后续真实执行的底座。

### 做错顺序的部分

- 还没有 live web search，却先做复杂人工盲审。
- 把人审做成全量必需步骤，导致项目无法由一个开发者持续迭代。
- 工作台暴露的是评测内部结构，不是用户真正需要的研究过程和报告。
- 当前十题合成语料适合单元/诊断测试，不足以代表市面上的 Deep Research。

## 6. 新项目定位

本项目不做“小型 DeerFlow”，也不训练模型。新的定位是：

> 面向 OpenAI-compatible 冻结模型的、预算可控且证据可审计的 Deep Research harness。重点不是让底座模型“变强”，而是让搜索停止、证据覆盖、引用回溯和失败定位变得可测量。

与同类项目相比，准备保留的独特性是：

1. **Evidence Debt**：计划中的每个回答义务必须链接到证据、明确保持 open，或触发下一轮搜索。
2. **Budget-aware stopping**：在 Token/费用/调用次数约束下，按照剩余 Evidence Debt 决定下一次搜索，而不是固定循环次数。
3. **Auditable trace**：每条最终 Claim 可以回溯到 evidence、URL、搜索轮次和成本事件。
4. **Bad-case queue**：自动评测失败直接形成可重放 case，用它决定是否需要 re-plan、critic 或 subagent。

## 7. 更简单的新 MVP

先只实现一个 Agent：

```text
Question
  -> Scope/Plan
  -> Search provider
  -> Fetch/read pages
  -> Evidence notebook + Evidence Debt
  -> 最多一次缺口驱动的追加搜索
  -> cited report
```

暂不实现：多 Agent、通用 Research DAG、无限反思、复杂人工审核平台。

用户实际看到的界面应只有：

- 输入问题、目标格式和可选来源范围；
- 查看/修改计划；
- 查看当前搜索进度和剩余预算；
- 阅读最终中文报告；
- 点击引用查看原网页与对应证据；
- 下载报告和 trace。

## 8. 新评测方案

### 第一层：每次运行的确定性检查

- 是否成功完成；
- 引用 marker 是否能解析到 evidence 和 URL；
- URL 是否抓取成功；
- Claim 是否有来源链接；
- Evidence Debt 是否显式 resolved/open；
- Token、费用、延迟、搜索轮数和失败类型。

### 第二层：自动语义评测

- 使用固定、版本化的中文 binary rubrics；
- Judge 输出每条 rubric 的 `pass/fail + reason + report evidence`；
- Claim-URL 支持度单独评测；
- Judge 模型、prompt、seed、输入哈希和费用全部留痕；
- 若 Judge 与生成模型相同，只能标为 calibration，不作为强结论。

### 第三层：外部 benchmark

- 先跑 5-10 个公开任务完成接口和成本校准；
- 再决定运行 GAIA、DeepResearch Bench 或 DeepResearch Bench II 的哪一部分；
- 同模型、同搜索工具、同预算比较 baseline 与 Evidence Debt；
- 未运行的数值全部标为 planned。

### 人工角色

人工审核不再是推进门槛。只在自动 Judge 出现分歧、引用抓取失败或需要形成求职案例时，抽查 3-5 个代表性 bad cases。用户不承担 40 份报告的逐 Claim 标注工作。

## 9. 下一阶段顺序与验收标准

1. **已完成 Live search/fetch 闭环**：DeepSeek 从中文问题生成带可点击官方来源的中文报告；trace 包含查询、URL、Token、费用和延迟。结果与失败链见 `live_web_smoke_2026-08-13.md`。
2. **已完成外部 benchmark baseline**：固定 5 个 LiveDRBench preview 任务，当前 no-key 搜索下兼容性 exact main-claim F1 为 0；官方 Judge 未运行。结果见 `livedrbench_preview_v0_results.md`。
3. **稳定搜索 provider**：在相同 query policy 和预算下替换通用 Bing RSS 回退，先修 first-pass retrieval；密钥仍只从环境变量读取。
4. **受控 Search A/B**：提前冻结独立 holdout，同模型、同查询、同 evidence cap、同 Token 或费用预算比较搜索 provider。
5. **延后单轮补搜**：只有稳定搜索仍留下可复现 evidence gap 时，才实现最多一轮 requery；不引入多 Agent。

只有在重复坏例证明单 Agent 边界不足后，才考虑并行 subagent、Critic-Repair 或 Research DAG。
