# 当前项目决策背景

- 定位：底座模型参数完全冻结，仅通过 OpenAI-compatible API 调用模型。报告只能描述 harness、搜索、证据和预算策略的效果，不能表述为模型能力提升。
- 已实现：单 Agent 的 Plan -> Search/Fetch -> Claim-Evidence Ledger -> cited report；Pydantic run state；查询、URL、Token、费用和延迟 trace；确定性引用组装；离线 fake provider 与测试。
- 当前搜索：仓库型查询优先使用 GitHub 公共仓库搜索，通用网页使用无密钥的 DuckDuckGo Lite/Bing RSS 回退。后两者是 best-effort，不是稳定搜索 SLA。
- 尚未实现：稳定的正式搜索 API、来源质量分层、迭代补搜/重规划、自动语义评测、Research DAG、多 Agent 和 Critic-Repair。
- 本次术语："评测方式"指 benchmark、任务集、指标或 judge protocol；LangSmith/Langfuse 等运行 trace 属于可观测性，不等同于效果评测。
- 决策要求：只推荐一个当前尚未实现、能由本次 bad case 直接证明必要的最小改进。优先简单方法，不因为外部项目使用多 Agent 就复制多 Agent。
