# LiveDRBench Fresh Public v0 注册（2026-08-17）

## 为什么现在要建新切片

BrowseComp-Plus 的 175 个 development 题已经全部生成和诊断；剩余 655 题是 sealed
holdout，不能把它们当作下一次机制调参的素材，也不能在已观察的 175 题中事后切一块伪装
成 fresh slice。

因此本项目固定一个**未使用的公开 LiveDRBench preview/test 切片**，只作为下一次外部
fresh public development diagnostic 的入口。它不会替代 BrowseComp-Plus 主 benchmark，
也不会把兼容性指标写成官方分数。

## 固定选择

已经使用的 preview keys 是 `[4, 31, 40, 55, 76]`。选择规则在生成前冻结：读取 pinned
revision 的全部十个公开 preview key，排除这五个旧 key，按整数升序取前五个。因此新切片为
`[10, 23, 38, 86, 99]`，LF-final-LF SHA256 为
`a4de10cb92e529184acbe76c6d1a28101d69f283eb05627be364ff00b41b1470`。

它覆盖 `entities`、两个 novel-datasets 子类和两个 `prior-art` 任务。选择只依赖公开 key
和既有切片，没有读取新答案、生成结果或 bad case。

## 预注册比较边界

| 项目 | 已固定内容 |
| --- | --- |
| 模型 | `deepseek-v4-flash`，thinking disabled |
| baseline | 既有 `public_benchmark.b1_benchmark_structured` live collector |
| candidate | `stable-search-provider-adapter-v0`，仍为 planned，尚未实现 |
| 每题上限 | 3 次模型调用、5 次搜索、8,000 Tokens、$0.002、6 条 evidence |
| evaluator | `compatibility_exact_main_claim_v1`；官方 evaluator 仍 planned |
| 数据边界 | BrowseComp-Plus sealed holdout 不访问 |

这一步只完成任务选择和比较契约。现有 public runner 还不能强制 5 次 search 上限，也没有
paired baseline/candidate execution wrapper；在这两个边界实现并测试之前，不能开始付费运行。
稳定 Search API 的选择、密钥和计费也尚未进入仓库。

## 验证

离线静态检查不会联网、不会调用模型：

```powershell
python scripts/check_livedrbench_fresh_public_registration.py `
  --registration benchmarks/livedrbench_fresh_public_v0/registration.json
```

只有显式传入 `--verify-pinned-dataset` 才会读取公开 Hugging Face pinned rows，并验证响应
hash、task key 与类别；它仍不调用 provider、Judge 或 GPU。

## 产物和边界

- registration：`benchmarks/livedrbench_fresh_public_v0/registration.json`
- contract：`src/deepresearch_harness/livedrbench_fresh_public.py`
- checker：`scripts/check_livedrbench_fresh_public_registration.py`
- tests：`tests/test_livedrbench_fresh_public.py`

注册并不是实验结果。没有生成、没有 candidate 实现、没有官方 Judge、没有效果数字，也没有
任何模型能力提升的主张。
