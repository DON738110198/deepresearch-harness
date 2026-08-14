# Evidence Bandwidth 确认实验：召回提升没有转化为答案正确率

## 结论

`Evidence Bandwidth Exchange v0` 未通过晋升门槛，决策为 `reject`。

它把同一冻结模型下的 evidence recall 从 `39.676667%` 提高到
`49.870000%`，增量为 `+10.193333 pp`，说明 dense ranks 6-20 中确实有
可用证据。但校准开发 Judge accuracy 从 `34.666667%` 降到
`26.666667%`，增量为 `-8.000000 pp`。搜索次数、总 Token 和 API 费用也
分别达到 baseline 的 `1.205934x`、`1.204590x` 和 `1.163563x`，超过
预注册上限。

因此，raw top-20 不是可晋升方案。观察到的是冻结模型上的 harness 效果，
不是模型能力提升或下降，也不是官方 leaderboard 分数。

## 问题链

### 1. 先定位问题

前一轮 fresh-25 Dense top-5 只比 BM25 提高 `3.636667 pp` evidence
recall，未达到 `+10 pp` 门槛。随后对已保存 query 做零 LLM 调用的 depth
probe：Dense top-20 相对 BM25 可提高 `13.113333 pp`，说明相关文档常在
rank 6-20，而不是 corpus 中不存在。

### 2. 先试更简单的方法

在不增加输出条数的前提下，测试了三种 fixed top-5 selector。最好的
BM25/Dense RRF 变体只提高 `6.31 pp`，仍未过门槛，因此没有直接增加
reranker、Critic、Research DAG 或多 Agent。

### 3. 最小假设

将 Dense top-20 暴露给 Agent，但把 20 条 snippet 的总预算限制为
`1792` tokenizer tokens。离线校准得到该 payload 为已存 BM25 top-5
payload 的 `0.933931x`。假设是：扩宽 evidence bandwidth 能提高召回，
同时端到端 Token 和费用不超过 baseline 的 `1.15x`。

## 冻结合同

| 项目 | 冻结值 |
|---|---|
| 数据 | BrowseComp-Plus development，排序后 offset 80，25 题 |
| 重复 | 3 次相同题集，baseline-first / candidate-first / baseline-first 交替 |
| 生成模型 | `deepseek-v4-flash`，参数冻结 |
| 控制策略 | `answer_reserve_nonthinking_v0` |
| baseline | BM25 top-5，每条最多 512 snippet tokens |
| candidate | Qwen3-Embedding-0.6B dense top-20，总 snippet 预算 1792 |
| Judge | Qwen3-32B BF16，固定 revision 和官方 grader prompt 合同 |
| sealed holdout | 未访问 |
| 最大 DeepSeek 费用 | `$2.50` |

预注册文件：`benchmarks/browsecomp_plus_v0/evidence_bandwidth_confirmation_v0.json`。

## 三轮结果

| 指标 | BM25 top-5 | EBX top-20 | 差值或比率 | Gate |
|---|---:|---:|---:|---|
| schema complete | 96.000000% | 98.666667% | +2.666667 pp | 通过 |
| strict exact，诊断 | 16.000000% | 17.333333% | +1.333333 pp | 非主 gate |
| evidence recall | 39.676667% | 49.870000% | +10.193333 pp | 通过 |
| gold recall | 41.566667% | 52.813333% | +11.246666 pp | 非主 gate |
| 校准 Judge accuracy | 34.666667% | 26.666667% | -8.000000 pp | 失败 |
| 每题搜索次数 | 12.693333 | 15.306667 | 1.205934x | 失败 |
| 每题总 Token | 246,917.106667 | 297,433.880000 | 1.204590x | 失败 |
| DeepSeek 费用 | $0.885546570 | $1.030389500 | 1.163563x | 失败 |
| 每题墙钟延迟 | 82,097.093333 ms | 96,439.360000 ms | 1.174699x | 记录项 |

合计 DeepSeek 费用为 `$1.915936070`。150 个生成运行全部成功，零最终
provider failure，零输出预算越界。150 个 Judge 结果全部解析成功，零请求
失败。evidence recall 的 query-trial 配对为 candidate 36 胜、baseline 20
胜、19 平；Judge 配对为 candidate 10 胜、baseline 16 胜、49 平。

## Bad Case 定位

Judge 判错的 75 个配对运行按保存的诊断 trace 分类：

| 错误类型 | BM25 | EBX |
|---|---:|---:|
| 格式不完整 | 3 | 1 |
| 没有召回相关文档 | 24 | 19 |
| 已召回相关文档但仍答错 | 22 | 35 |
| 总错误 | 49 | 55 |

EBX 确实减少了 5 个 no-evidence failure，却新增了 13 个
relevant-evidence-but-wrong failure。16 个 candidate regression 中，8 个在
candidate 侧已有相关证据，3 个甚至比对应 baseline 有更高 evidence
recall。反过来，10 个 candidate improvement 中有 9 个伴随更高 evidence
recall。

边界明确：更宽的候选证据既有价值，也会带来噪声、重复搜索和合成负担。
当前证据支持“召回增益没有稳定转化为答案正确率”，但还不能单独证明原因
一定是上下文长度；也可能包含候选排序、证据冲突或停止策略问题。

## 零成本 Selectivity Probe

随后按单独注册的
`benchmarks/browsecomp_plus_v0/evidence_selectivity_probe_v0.json` 重放全部
75 个配对 trace。该步骤只读取保存的搜索结果、prediction-bound
development docid 和 Judge label，`provider_calls = 0`。

| 配对结果 | 数量 | baseline 找到相关证据 | candidate 找到相关证据 | baseline / candidate 搜索次数 | candidate 重复 doc slot 率 |
|---|---:|---:|---:|---:|---:|
| candidate improvement | 10 | 30.00% | 100.00% | 12.40 / 19.00 | 49.77% |
| candidate regression | 16 | 100.00% | 50.00% | 14.75 / 12.50 | 47.07% |
| both correct | 10 | 100.00% | 100.00% | 9.90 / 25.30 | 55.29% |
| both incorrect | 39 | 48.72% | 69.23% | 12.64 / 12.95 | 38.58% |

这个结果修正了“只要减少 top-20 噪声即可”的过早判断：

- improvement 组显示 dense 长尾是必要增量，candidate 将相关证据命中率从
  30% 提到 100%；
- regression 组显示不能用 dense 完全替换 BM25，baseline 侧 100% 找到相关
  证据，而 candidate 只有 50%；
- both-correct 组中 candidate 平均搜索 25.3 次、总 Token 574,894，对应
  baseline 的 9.9 次和 140,631 Token，存在明显的过度研究；
- 所有组的重复 query string 率均为 0%，但同一运行中重复 doc slot 率为
  38.58% 到 55.29%，问题是不同 query 重复灌入相同文档，而不是原样重复
  query。

因此下一候选不是纯 dense progressive disclosure，而是双通道设计：保留
BM25 top-5 作为高精度 anchor；只把去重后的 dense 长尾作为短 lead；模型
显式选择后才能 `open_evidence`。这同时针对 regression、long-tail rescue 和
重复 evidence ingress，仍然不需要多 Agent。

## 下一阶段

下一候选机制暂定为双通道 `Evidence Progressive Disclosure`，而不是多
Agent：BM25 top-5 返回完整 anchor，dense 长尾只返回去重后的短 lead；
Agent 只有显式选择后才能打开完整文档。目标是把“发现候选证据”与“把
证据写入上下文”拆开，保留长尾召回，同时限制噪声和累计 Token。

实现与验收顺序：

1. **零成本 trace probe，已完成**：75/75 配对观察，`provider_calls = 0`，
   全部 source SHA-256 通过；结论见上节。
2. **离线工具合同，下一步**：实现 BM25 anchor、dense `search_leads` 和受限 `open_evidence`，每次打开
   和累计 evidence-ingress Token 都写入 trace。验收要求是 deterministic
   fixture 测试、非法 docid fail closed、密钥仍只从环境变量读取。
3. **fresh-5 工程 smoke**：只验证协议、预算、引用和恢复，不作效果声明。
   要求 5/5 完成、零越界、零未审计 evidence open。
4. **fresh-25 预注册确认**：模型、prompt、语料、Judge、执行顺序不变；
   candidate 相对 BM25 仍须 evidence recall `>= +10 pp`、Judge accuracy
   `>= 0 pp`、搜索次数比 `<= 1.10x`、总 Token 和费用比 `<= 1.15x`，并新增
   relevant-evidence-but-wrong 不得高于 baseline 的 gate。
5. **只在通过后**冻结方案并考虑 sealed holdout。若再次出现相关证据已到位
   但答案退化，则优先研究结构化 Evidence Packet / Claim-Evidence Ledger
   的选择与核验；仍不以增加 Agent 数量代替诊断。

## 产物

- 运行根目录：`runs/browsecomp_plus_v0/repeats/evidence-bandwidth-confirmation-v0-fresh25-20260814/`
- 自动聚合：`repeat_comparison.json`
- Judge：`persistent_judge_v0/evaluation/repeat_development_judge_comparison.json`
- 机器决策：`evidence_bandwidth_decision.json`
- 决策 SHA-256：`981fcd2a6ee1f65bf6f48823a505fdcc7ab540f3aa9f7445bd8a09e6ae3647b8`
- Selectivity probe：`evidence_selectivity_probe.json`
- Probe SHA-256：`1a822d5f20e0139bc40ece1fc150dcd89a0b7d7bdd559970e71f58d9aa2514e3`

所有 `runs/` 产物保留在本地审计目录，不提交 API key；Git 中提交预注册、
实现、测试和本结果说明。
