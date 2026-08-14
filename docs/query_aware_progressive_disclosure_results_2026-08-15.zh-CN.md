# Query-Aware Progressive Disclosure 实验记录

## 结论

这轮实验把一个宽泛想法收敛成了可审计的因果链：

```text
宽检索带来重复上下文和错误综合
-> BM25 anchor + Dense lead + 显式 open
-> 工具循环失控
-> 每题最多 8 次搜索
-> Dense lead 预览不可选择
-> query-aware passage preview
-> fresh-25 配对确认
```

fresh-25 上，候选 harness 的校准 Judge accuracy 从 `16%` 到 `20%`，
evidence recall 从 `36.03%` 到 `44.79%`，搜索、总 Token 和 DeepSeek API
费用比分别为 `0.488764`、`0.251863` 和 `0.483486`。但预注册要求两个
variant 的 schema completeness 都至少为 `96%`，实际 baseline 只有
`23/25 = 92%`，因此 24 项合取门槛的机器决策必须是 `reject`。不能因为
其余 23 项通过，就在看到结果后删除这个失败项。

这不是官方 BrowseComp-Plus accuracy、sealed-holdout 结果或排行榜成绩，
也不是冻结模型能力提升。它只是同一 DeepSeek V4 Flash 下的一次开发集
harness/orchestration 对比。

## 为什么做这个机制

前一轮 Evidence Bandwidth top-20 虽然提高了 evidence recall，却降低了
Judge accuracy，并显著增加搜索、Token 和费用。保存 trace 显示两个问题：
Dense 能补到长尾证据，但会丢失 BM25 anchor；不同搜索还会反复把相同文档
送回上下文。因此，更简单的下一步不是增加 Agent，而是改变证据接口：

- BM25 top-5 作为完整 anchor 保留；
- Dense-only top-15 只返回短 lead；
- Agent 认为 lead 有价值时再显式 `open_evidence`；
- 所有搜索、open 和 evidence ingress 都写入 trace。

## 逐步实验

| 阶段 | 观察 | 决策 |
| --- | --- | --- |
| Progressive Disclosure fresh-5 | Judge `2/5 -> 2/5`；搜索 `32 -> 65`；Token 比 `3.131361`；费用比 `1.218099` | `reject`，先限制循环 |
| Tool-Loop Governor fresh-5 | evidence recall `20% -> 43.33%`；Token 比 `0.195385`；费用比 `0.445916`；Judge 仍为 `0/5 -> 0/5` | `reject`，资源改善不能替代答案质量 |
| Query-aware preview 离线校准 | 相关 Dense lead 的可选择预览 `0/4 -> 4/4`；provider call 为 `0` | 允许 fresh-5 |
| Query-aware preview fresh-5 | Judge `3/5 -> 3/5`；evidence recall `55.833333% -> 80%`；Token 比 `0.324788`；费用比 `0.536012` | `promote` 到 fresh-25 |
| Query-aware preview fresh-25 | 见下一节 | 总门槛 `reject` |

每一阶段都使用新的注册文件和问题切片；失败结果没有被覆盖，也没有用后续
结果反向修改旧阈值。

## Fresh-25 结果

控制合同：同一 `deepseek-v4-flash`、`high` thinking、空 system prompt、
10k 全局输出上限和 `answer_reserve_nonthinking_v0`。baseline 是 Pi v8 +
BM25 top-5；candidate 是 Pi v10 + 5 个 BM25 anchors + 15 个 query-aware
Dense leads + 最多 8 次搜索/8 次 open。25 个问题来自此前未评测的
development ID，生成前已冻结；sealed holdout 未访问。

| 指标 | Baseline | Candidate | 差异或比例 |
| --- | ---: | ---: | ---: |
| 成功 / 失败 / budget exhausted | 25 / 0 / 0 | 25 / 0 / 0 | 通过 |
| schema complete | 23/25，92% | 24/25，96% | baseline 门槛失败 |
| calibrated Judge correct | 4/25，16% | 5/25，20% | `+4 pp` |
| strict exact | 3/25，12% | 4/25，16% | `+4 pp` |
| evidence recall | 36.03% | 44.79% | `+8.76 pp` |
| gold recall | 40.00% | 44.47% | `+4.47 pp` |
| search calls | 356 | 174 | `0.488764x` |
| total Tokens | 7,357,600 | 1,853,106 | `0.251863x` |
| DeepSeek API cost | $0.34142066 | $0.16507202 | `0.483486x` |
| combined generation cost | - | - | $0.50649268 |
| output overshoot | 0 | 0 | 通过 |
| Judge parse/request failures | 0 / 0 | 0 / 0 | 通过 |

24 个预注册 gate 中只有 `baseline_schema_complete_percent >= 96` 失败。
机器决策及其逐题 trace 位于忽略目录
`runs/browsecomp_plus_v0/query-aware-preview-confirmation-v0-20260815/paired25-decision.json`。
该文件 hash 绑定注册、问题、两个 summary、diagnostic 和 Judge 结果。

## Bad-case 诊断

配对 Judge 结果为：candidate improvement `3`、regression `2`、双方都对
`2`、双方都错 `18`。三个 improvement 中，baseline 的 evidence recall
都为 `0`，candidate 分别达到 `0.75`、`0.90` 和 `1.00`。两个 regression
则从 baseline 的 `1.00` 降到 candidate 的 `0.14` 和 `0.33`。

错误层也发生了移动：

| 错误层 | Baseline 错误 21 题 | Candidate 错误 20 题 |
| --- | ---: | ---: |
| zero evidence recall | 11 | 7 |
| evidence present but Judge wrong | 10 | 13 |

这说明 Query-Aware Progressive Disclosure 确实减少了一部分零召回，但
主要剩余问题已经变成“召回了部分相关证据，却没有稳定验证候选实体和答案”。
保存 trace 中，一个 regression 在 8 次搜索后仍明确承认关键条件未确认却
输出低置信候选；另一个只搜索 4 次便对错误运动员给出高置信答案，并引用了
不能直接支持年份的证据。它们不是“再加几个 Researcher”能直接解决的
问题，而是未解决 evidence debt 在最终编译前没有成为硬门槛。

17/25 个 candidate run 触发了 8 次搜索上限，只有 2 个额外搜索调用被拒绝；
因此 governor 主要消除了无界循环，并没有造成大规模工具调用异常。显式
open 成功 6 次，说明短预览本身已经能承担多数选择，后续不应简单扩大 open
预算。

## 下一阶段

下一机制预注册为单 Agent 的 **Evidence-Debt Search Reserve**，而不是 B3/B4
多 Agent：

1. 先在保存 trace 上做零搜索、零 provider-call 的 debt 标注器测试。输入为
   原问题、候选答案和已有证据；输出必须是结构化 obligation、support 状态和
   最多两条 repair query。它至少要标记两类已观察失败：明确自述“关键条件未
   确认”，以及结论所依赖年份/身份没有直接证据。
2. 若离线合同通过，实现 Pi v11：最多 6 次探索搜索，预留 2 次 debt-guided
   repair；探索 8k、debt audit 512、最终 compiler 1488，总输出上限仍为
   10k，总搜索上限仍为 8。共享 Claim-Evidence Ledger 仍是唯一事实源。
3. 先做 outcome-selected bad-case calibration，只能称为定向诊断。验收标准是
   修复至少一个已知 regression，且不破坏三个已知 improvement；该结果不能
   用作 benchmark claim。
4. 只有定向诊断通过，才在全新 development slice 上比较 v10 与 v11。固定
   模型、retriever、问题数、8 次搜索、10k 输出和 Judge；要求 candidate
   schema completeness 至少 96%、Judge 不下降、evidence recall 不下降、
   Token 和费用比各不超过 1.15、零解析/请求失败。
5. 新 fresh-25 通过后才做三次完整重复。sealed holdout、官方 evaluator 和
   leaderboard submission 仍为 `planned`。

B3 固定三 Researcher 和 B4 动态 Sub-Agent 继续保持 `planned`。只有重复出现
独立分支遗漏、单上下文干扰、矛盾证据未核验或串行延迟成为主瓶颈时，才允许
注册多 Agent 对比。

## 当前限制

- 只有一次 25 题开发集配对，不能估计跨 trial 方差。
- 当前 Judge 是已校准的 Qwen3-32B BF16 常驻服务，不是上游官方执行适配器。
- evidence recall 只表示相关文档被召回，不等于证据足以支持最终答案。
- 本轮没有 citation support、冲突发现率或人民币成本；这些指标仍为
  `planned`，不得补写估计值。
