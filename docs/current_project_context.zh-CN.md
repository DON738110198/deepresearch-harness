# 当前项目决策背景

- 定位：底座模型参数完全冻结，仅通过 OpenAI-compatible API 调用模型。报告只能描述 harness、搜索、证据和预算策略的效果，不能表述为模型能力提升。
- 已实现：单 Agent 的 Plan -> Search/Fetch -> Claim-Evidence Ledger -> cited report；Pydantic run state；查询、URL、Token、费用和延迟 trace；确定性引用组装；离线 fake provider 与测试。
- 当前搜索：baseline 的仓库型查询优先使用 GitHub 公共仓库搜索，通用网页使用无密钥的 DuckDuckGo Lite/Bing RSS 回退；它们是 best-effort，不是稳定搜索 SLA。候选的 Tavily basic `/search` adapter 已实现，answer/raw content 均关闭，密钥只从 `TAVILY_API_KEY` 环境变量读取。
- 尚未实现：来源质量分层、迭代补搜/重规划、自动语义评测、Research DAG、多 Agent 和 Critic-Repair。Fresh public 的 hash-bound paired executor 与显式 failed-only resume 已实现但尚未执行。
- 本次术语："评测方式"指 benchmark、任务集、指标或 judge protocol；LangSmith/Langfuse 等运行 trace 属于可观测性，不等同于效果评测。
- 决策要求：只推荐一个当前尚未实现、能由本次 bad case 直接证明必要的最小改进。优先简单方法，不因为外部项目使用多 Agent 就复制多 Agent。
- 最新诊断：175 题 development profile 的 110 个 calibrated-Judge-wrong trace 已完成 gold/evidence 纠偏。40 个无 reference、21 个仅 supporting evidence、31 个 gold span 不完整、8 个可见未引、7 个已引仍错、3 个 answer contract。它只是 selected-after-v0 的 queue map；下一次付费机制测试必须用 fresh development slice，global span opening 和多 Agent 都暂不进入实现。
- Fresh public 入口：LiveDRBench 已固定未使用 keys `[10,23,38,86,99]`。baseline 是既有 live collector；candidate 是 Tavily basic search adapter。现可离线冻结两臂 config snapshot，并通过 executor 执行或仅失败恢复；当前仍没有生成、官方 Judge 或效果数字。搜索 attempt 硬上限为 5；这个比较只 token-match，工具费用必须单列，不能冒充 cost-match。
