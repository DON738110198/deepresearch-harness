# Persistent Miss Dense Rank v0 结果

## 问题

同一批 7 个 persistent retrieval miss 已通过语料可回答性检查：18/18 gold 文档
存在，7/7 题的 literal answer 位于 gold 文档中。全文 BM25 top-20 为 `0/7`，
完整 passage BM25 collapsed top-20 为 `2/7`。本轮只回答一个更小问题：冻结的
Qwen3-Embedding-0.6B 是否已经把 gold 文档放进可用的候选深度。

## 预注册

注册提交 `23078f1` 先于实现和结果进入 `main`。固定项包括：

- 相同 7 题与 trial-1 的 48 条成功 search query，顺序和 SHA256 均复核；
- 100,195 文档的同一 dense index；
- Qwen3-Embedding-0.6B revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；
- query prefix、EOS pooling、L2 normalization、max length 512；
- depth `20, 100, 1000`，batch size 8，CPU execution；
- top-20 至少 `4/7` 才能选择 dense candidate；若它失败，top-100 至少 `4/7`
  才能进入 bounded reranker 诊断；
- provider、在线搜索、Judge 均为 0；sealed holdout 禁止。

这是 outcome-selected、gold-aware rank diagnostic，不是 end-to-end、官方 accuracy、
leaderboard 或模型能力结果。

## 执行完整性

第一次本机 CPU 尝试没有生成任何结果。Windows Application Error/WER 记录
`torch_cpu.dll` 异常 `0xc0000005`；当时本机只有约 1.49 GiB 可用内存，并且两个
保留的 dense retriever 已各自装载模型。内存压力是合理解释，但不是已证明根因，
所以没有在同一环境盲目重试，也没有停止 8768/8769 或 Judge 服务。

实现提交 `338f3ab` 通过 193 个 Python 测试后推送。随后把同一提交、byte-verified
模型和四个 index shard 复制到高内存服务器，显式设置空
`CUDA_VISIBLE_DEVICES` 后用 CPU 完成运行。artifact 内的 `runtime.device=cpu`；
PyTorch 包含 CUDA build 字样不代表使用了 GPU。远端与拉回本地的 artifact SHA256
均为 `0ceeddf7fb1e54fb94c90f4920e577ae71917ed7007d02cd1681c7c226a97bb6`。
恢复过程另存于 `runs/browsecomp_plus_v0/persistent-miss-dense-rank-v0-20260816/execution.json`，
SHA256 为 `d1e1f8e495757dc9281f68e814dcd2bce7358560f4c3a07c02d0c1f3247d4737`。

## 结果

| Depth | Gold-hit cases | Threshold | Pass |
| --- | ---: | ---: | --- |
| top-20 | 0/7 | >=4/7 | no |
| top-100 | 1/7 | >=4/7 | no |
| top-1000 | 5/7 | diagnostic only | n/a |

| Query ID | Best dense gold rank | Passage top-20 hit |
| --- | ---: | --- |
| 875 | 109 | no |
| 754 | 981 | yes |
| 898 | missing at 1000 | no |
| 710 | 551 | no |
| 873 | 78 | no |
| 869 | missing at 1000 | yes |
| 805 | 211 | no |

Dense top-20 相对 passage top-20 是 `-2` 个 case，wins 为 0、losses 为 2。索引
加载 39,040 ms，48 条唯一 query 的 dense search 为 52,199 ms。provider、在线
search、Judge call 均为 0。

机器决策为 `freeze_dense_channel`。不能因为 top-1000 有 `5/7` 就在观察结果后把
门槛改成 1000 或追加大池 reranker；预注册只允许 top-20 candidate 或 top-100
pool diagnosis，两者都失败。

## 框架边界

- [DeerFlow](https://github.com/bytedance/deer-flow) 的动态 Sub-Agent 可扩大独立研究
  分支，但共享 dense top-20 为 `0/7` 时，增加 worker 仍看不到当前目标文档。
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
  的可配置 search backend 提醒我们保持检索边界可替换；本轮结果只拒绝这个固定
  query/index/candidate-depth 组合，不否定所有 dense retrieval。
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 的 crawler/execution
  agents 适合扩大外部搜索，本轮尚未证明增加调用比定位缺失 bridge 更简单。
- [MindSearch](https://github.com/InternLM/MindSearch) 的搜索图适合从可见节点扩展
  新分支；下一步先测一个确定性的一跳 visible pivot，而不是直接实现多 Agent 图。

## 下一步

该 Visible-Pivot Bridge Sufficiency oracle 已完成并以 `4/7` 刚好通过。它只授权
设计一个新的 gold-blind pivot selector；其中 q875 的成功 token `inlin` 来自坐标
frontmatter 中的格式词 `inline`，所以 selector 必须先删除 wrapper/frontmatter，
不能把 oracle pass 写成语义检索改进。完整结果见
`docs/visible_pivot_bridge_results_2026-08-16.zh-CN.md`。fresh paired run、Judge
accuracy、citation support、Token、费用和多 Agent 仍保持 `planned`。
