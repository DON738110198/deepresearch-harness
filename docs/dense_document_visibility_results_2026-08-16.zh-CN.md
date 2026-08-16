# Dense Document Visibility 结果（2026-08-16）

## 问题纠正

上一轮把 retriever manifest 中的 `max_length=512` 同时解释成 query 和 document
限制。源码核对后确认这不成立：BrowseComp-Plus 官方复现命令对 query 使用
`query_max_len=512`，对 document 使用 `passage_max_len=4096`，document prefix 为空，
不追加 EOS。Tevatron 的冻结 collator 对 `text.strip()` 做 right truncation，并保留
`add_special_tokens=true`。

同时发现一个独立的可审计性问题：下载的预构建 index revision
`b3f37f70c33829eb09d04784a54277a31871fd63` 只发布向量分片和简短 README，没有把
model revision、tokenizer hash、query/document 长度与 preprocessing 绑定到向量。
因此本轮只能称为 **官方复现 recipe 下的可见性诊断**，不能反推历史向量的精确输入。

## 预注册

- failure cluster：相同 7 个 persistent dense retrieval miss、18 个 gold 文档；
- 输入：固定 Qwen3-Embedding-0.6B tokenizer 文件，document window `4096`；
- document visible：tokenizer encode-decode 后仍包含全部 normalized literal answer atoms；
- case visible：至少一个 gold document visible；
- gate：visible cases `>=4/7` 时拒绝“head truncation 是主要原因”；
- 调用预算：provider、embedding model、search、Judge、GPU 全部为 `0`；
- sealed holdout：禁止。

## 结果

| Query | Visible gold docs | Gold docs | Truncated docs | Case visible |
| --- | ---: | ---: | ---: | --- |
| 875 | 8 | 8 | 3 | yes |
| 754 | 1 | 1 | 0 | yes |
| 898 | 1 | 1 | 0 | yes |
| 710 | 1 | 1 | 0 | yes |
| 873 | 1 | 1 | 1 | yes |
| 869 | 3 | 4 | 3 | yes |
| 805 | 2 | 2 | 1 | yes |

汇总为 visible cases `7/7`、visible documents `17/18`、truncated documents `8/18`。
唯一隐藏 answer 的文档是 q869 的 `34297`，但该题另有 3 个 answer-visible gold
文档，所以 case 仍可见。预注册 `4/7` gate 通过，decision 为
`reject_head_truncation_hypothesis`。

正式执行前保留了两个零调用失败：`uv` 首次自动选中 Python 3.10，间接依赖没有
cp310 wheel；重建 Python 3.12 环境后，第一次脚本执行又因远端缺少注册过的 ignored
prerequisite artifact 而在 document read 前 fail-closed。同步并核对已有 artifact SHA
后，同一 registration 成功完成。没有重跑任何计费调用。

## 结论

`4096-token` head truncation 不是这 7 题 dense miss 的主要解释，不能据此投入
passage-dense index。更小的下一问题是：48 条 Agent 生成查询是否丢失了原问题中的
组合约束，导致 query/document semantic alignment 变差。下一轮只比较每题原始完整
question 的 dense rank 与已冻结 generated-query best rank；仍不加 Agent、reranker、
provider 或 Judge。

这个结果不是 retrieval improvement、端到端准确率、官方 benchmark 或模型能力提升。

## 产物

- registration：`benchmarks/browsecomp_plus_v0/dense_document_visibility_v0.json`
- implementation：`src/deepresearch_harness/dense_document_visibility.py`
- runner：`scripts/run_dense_document_visibility.py`
- tests：`tests/test_dense_document_visibility.py`
- ignored audit：`runs/browsecomp_plus_v0/dense-document-visibility-v0-20260816/audit.json`
- audit SHA256：`18aa7a7e7d6913517affbc1db9b3b0df5203fc377fd9a59363f9e25c1a8ed8d2`
