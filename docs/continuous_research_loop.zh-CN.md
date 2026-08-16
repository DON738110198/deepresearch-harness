# 受控 Deep Research 研究循环

## 目的

这个项目不以“功能越来越多”为进度，而以一条可反驳的实验链为进度：

```text
定位一个主失败簇
-> 检查更简单的 baseline
-> 对照现有 Deep Research 框架
-> 只冻结一个机制变化
-> 离线校准
-> 新开发集实跑
-> Judge + 资源门槛
-> 接受或保留负结果
-> 再进入下一轮
```

每次准备暂停、换方向或宣布阶段完成前，都必须生成并验证一个
`ResearchLoopCheckpoint`。只要 loop 没有 measured result、机器 decision，
或者仍有付费调用在途，检查器都会返回 `ready_to_pause=false` 和明确的下一动作。

## 每轮边界

1. 一轮只允许改变一个机制；prompt、检索器、模型或预算的联动修改必须拆开。
2. 必须保留最简单 baseline，并写清为什么当前机制比“多搜几次”或“多加一个
   Agent”更适合已观察失败。
3. 至少对照三个当前开源框架，而且只能引用官方仓库或官方文档。对照内容必须
   是机制和失败边界，不比较 stars，也不把功能清单当创新。
4. 外部机制只有映射到保存的 bad case 才能进入实现。仅达到 DeerFlow、Open
   Deep Research、GPT Researcher 或 MindSearch 的功能平价不计作贡献。
5. 每轮在注册文件中固定模型、工具、语料、问题切片、搜索数、Token、费用上限、
   Judge 和停止规则。`sealed holdout` 始终禁止，直到三次开发集稳定确认通过。
6. outcome-selected bad case 只能用于机制校准，不能用于效果数字；效果判断必须
   使用此前未评测的新 development slice。
7. 失败实验不删除、不调阈值、不挑子集。下一轮只能从新的失败定位开始。

## 防抖与执行完整性

1. 同一机制层连续三次注册实验被拒绝后，该层默认冻结；更严格的注册可以提前
   冻结某个窄分支。本轮 terminal-verifier prompt 分支在 v0、v1 两次拒绝后已按
   checkpoint 约定提前冻结。
2. 冻结不是“以后永远不能碰”，而是必须先出现新的 failure cluster 或新的离线
   证据，不能继续在原结果上换措辞、调阈值或扩大样本来追阳性。
3. 搜索服务必须在每次 provider 子进程前通过 health 与精确 `retriever_id`
   检查。任一 search/open transport error 都使该题失败并中止批次；有流畅终稿
   也不能把工具失败改写成成功。
4. 失败进程和部分结果必须保存。恢复只读取已落盘的 per-case artifact；付费结果
   不得因进程中断而重复调用。
5. 离线 oracle、smoke 与静态测试只证明可用性或执行完整性，不得写成效果提升。

## 外部对照的作用

- DeerFlow 2.0 的动态 Sub-Agent、隔离上下文和终止条件适合真正的并行分支问题；
  当前错误不是这个类型，因此暂缓多 Agent。
- LangChain Open Deep Research 已把早期 supervisor-researcher 多 Agent 版本列为
  低于当前实现的 legacy。它提醒我们角色数量不是质量保证，但其压缩阶段和
  Deep Research Bench 导出边界值得后续对照。
- GPT Researcher 的 Planner、execution/crawler agents 和 publisher 强在搜索
  广度与报告生产；本项目要验证的是更小的答案级证据债务控制能否用更少资源
  达到更可靠的停止。
- MindSearch 用 WebPlanner 动态扩展搜索图并派发 WebSearcher，适合独立子问题
  漏检。只有我们的 bad case 反复出现独立分支遗漏时，才进入该层对比。

## 当前结论与方向

项目不会声称在通用产品能力上超过这些成熟框架。已经跑出的结论是：

- 全局暴露 span-open 工具会扰动搜索轨迹，两个完整 paired trial 合计少 4 个
  Judge-correct、少 4 个 strict exact，mean evidence recall 下降 8.92pp，因此冻结。
- 三次 v10 baseline 把 25 题分成 9 个 stable-correct、7 个 persistent retrieval
  miss、4 个 gold-doc-present/span-missing，以及 5 个其他合成或不稳定案例。
- 对 4 个 span-missing 案例，section-aware selector 的 gold-aware 离线校准达到
  4/4；no-gold target slate 也能在最多 4 个候选中达到 4/4。这只是可用性门槛。
- bounded post-run overlay v0 仅修复 2/4；隐藏 baseline 的 v1 为 0/4。两轮均被
  注册门槛拒绝，terminal-verifier prompt 分支冻结。
- 7 个 persistent retrieval miss 的 18 个 gold 文档全部在索引中，且 7/7 全文
  含 literal answer。raw full question 的 BM25 top-5 为 0/7；模型生成查询最好也
  只有 0/7 top-20、2/7 top-100、6/7 top-1000。

因此当前差异点不再表述为泛化的 `repair_search`，而是 **Evidence Debt 驱动的
检索表示诊断**：先证明 obligation、目标文档、答案 span 分别在哪一层丢失，再
决定是否改变 passage index、candidate pool 或验证器。下一候选是离线
passage-level/index-representation gate；多 Agent、fresh paired run、sealed holdout、
official Judge 和 leaderboard submission 都保持 `planned`。

## 操作命令

```powershell
python scripts/check_research_loop_checkpoint.py `
  experiments/evidence_debt_search_reserve_v0/checkpoint.json
```

日常检查允许 active loop 返回非暂停就绪；真正结束阶段时加
`--require-pause-ready`，未闭环会以非零状态退出。
