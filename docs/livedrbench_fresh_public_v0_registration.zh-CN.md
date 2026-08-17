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
| candidate | `tavily-basic-search-adapter-v0`；仅替换 search provider，已实现、尚未运行 |
| 每题上限 | 3 次模型调用、5 次实际 HTTP 搜索尝试、8,000 Tokens、$0.002 LLM、6 条 evidence |
| Tavily 工具预算 | `basic`、每次 1 credit、最多 5 credit；价格快照为 2026-08-17 的 $0.008/credit，即最多 $0.04/题 |
| evaluator | `compatibility_exact_main_claim_v1`；官方 evaluator 仍 planned |
| 数据边界 | BrowseComp-Plus sealed holdout 不访问 |

候选实现固定为 Tavily `/search` 的 `basic` depth：不接收 Tavily answer、不接收 raw content、
不启用图片；结果只会以 `title/url/content` 进入既有的受限抓取和 Ledger。密钥只读取
`TAVILY_API_KEY` 环境变量，不进入配置、trace 或注册文件。`LiveWebCollector` 现在在每次
实际 HTTP 搜索前扣一次共享预算，失败和异常也计数，第六次会在网络请求前拒绝；trace 会记录
逻辑 query 序号、HTTP attempt 序号、query hash、结果数、预算前后、延迟和估算工具费用。

这是 **token-matched 的搜索后端消融**，不是总成本匹配比较：baseline 无密钥，candidate 有
Tavily credit，因此 LLM 费用和 Tavily 费用必须分别报告，不能把结果写成 cost-matched 胜负。
当前仍缺少 hash-bound 的 paired runner，尚未生成、调用 Tavily、调用模型或运行官方 Judge。
执行前还必须重新核对 [Tavily credits 文档](https://docs.tavily.com/documentation/api-credits) 的
价格快照；若价格变化，应重新注册新的比较块，不能静默沿用 $0.008。

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

注册并不是实验结果。没有生成、没有 paired 比较、没有官方 Judge、没有效果数字，也没有任何
模型能力提升的主张。
