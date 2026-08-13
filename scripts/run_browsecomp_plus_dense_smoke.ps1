param(
    [int]$Port = 8766,
    [string]$Queries = "runs\browsecomp_plus_v0\dev_queries_5.json",
    [string]$OutputDir = "runs\browsecomp_plus_v0\pi_flash_dense_smoke",
    [ValidateSet("deepseek-v4-flash", "deepseek-v4-pro")]
    [string]$Model = "deepseek-v4-flash",
    [ValidateSet("standard", "answer_reserve_v0", "answer_reserve_v1", "answer_reserve_nonthinking_v0", "first_tool_deadline_v0", "tool_bootstrap_v0", "rare_anchor_portfolio_v0")]
    [string]$ControlPolicy = "answer_reserve_nonthinking_v0",
    [string]$Node = "",
    [switch]$ResumeFailed
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Manifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\target_manifest.json"
$RetrieverManifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\retriever_candidates.json"
$Partitions = Join-Path $Root "benchmarks\browsecomp_plus_v0\query_partitions.json"
$IndexRoot = Join-Path $Root "runs\_external\browsecomp-plus-indexes-direct"
$DocumentIndex = Join-Path $IndexRoot "bm25"
$ModelDir = Join-Path $Root "runs\_external\models\Qwen3-Embedding-0.6B"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$JavaHome = (Resolve-Path (Join-Path $Root "runs\_external\java21\jdk-21.0.12+8")).Path
$SnippetTokenizer = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
$Adapter = Join-Path $Root "integrations\pi-browsecomp"

if (-not $Node) {
    $BundledNode = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
    if (Test-Path -LiteralPath $BundledNode) {
        $Node = $BundledNode
    } else {
        $NodeCommand = Get-Command node -ErrorAction SilentlyContinue
        if ($NodeCommand) {
            $Node = $NodeCommand.Source
        }
    }
}

function Stop-ProcessTree {
    param([int]$RootProcessId)

    $Children = @(Get-CimInstance Win32_Process `
        -Filter "ParentProcessId = $RootProcessId" -ErrorAction SilentlyContinue)
    foreach ($Child in $Children) {
        Stop-ProcessTree -RootProcessId $Child.ProcessId
    }
    Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

if (-not [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "Process")) {
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
}
if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is not set in the process or user environment."
}
foreach ($Path in @(
    $Manifest,
    $RetrieverManifest,
    $Partitions,
    $IndexRoot,
    $DocumentIndex,
    $ModelDir,
    $Python,
    $JavaHome,
    $SnippetTokenizer,
    $Adapter,
    $Node
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing runtime artifact: $Path"
    }
}
$NodeVersion = [version](& $Node -p "process.versions.node")
if ($NodeVersion -lt [version]"22.19.0") {
    throw "Pi 0.84.1 requires Node.js >=22.19.0; resolved $NodeVersion at $Node"
}

$env:JAVA_HOME = $JavaHome
$env:Path = "$JavaHome\bin;$env:Path"
$ServerOut = Join-Path $Root "runs\browsecomp_plus_v0\dense-server.stdout.log"
$ServerErr = Join-Path $Root "runs\browsecomp_plus_v0\dense-server.stderr.log"
$Server = Start-Process -FilePath $Python -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $ServerOut -RedirectStandardError $ServerErr `
    -ArgumentList @(
        "-m", "deepresearch_harness.dense_server",
        "--manifest", $Manifest,
        "--retriever-manifest", $RetrieverManifest,
        "--candidate-id", "qwen3-embedding-0.6b",
        "--model-dir", $ModelDir,
        "--index-root", $IndexRoot,
        "--document-index-path", $DocumentIndex,
        "--snippet-tokenizer-dir", $SnippetTokenizer,
        "--port", $Port
    )

try {
    $Ready = $false
    for ($Attempt = 1; $Attempt -le 180; $Attempt++) {
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($Health.status -eq "ok") {
                $Ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $Ready) {
        throw "Dense server did not become ready. Inspect $ServerErr"
    }

    $RunArguments = @(
        "-m", "deepresearch_harness.cli", "run-pi-browsecomp-smoke",
        "--manifest", $Manifest,
        "--partitions", $Partitions,
        "--queries", (Join-Path $Root $Queries),
        "--output-dir", (Join-Path $Root $OutputDir),
        "--node", $Node,
        "--adapter-dir", $Adapter,
        "--search-url", "http://127.0.0.1:$Port/search",
        "--model", $Model,
        "--control-policy", $ControlPolicy,
        "--retriever-id", "qwen3-embedding-0.6b",
        "--retriever-manifest", $RetrieverManifest,
        "--timeout-seconds", "900"
    )
    if ($ResumeFailed) {
        $RunArguments += "--resume-failed"
    }
    & $Python @RunArguments
    if ($LASTEXITCODE -ne 0) {
        throw "BrowseComp-Plus dense smoke returned exit code $LASTEXITCODE"
    }
} finally {
    if ($Server -and -not $Server.HasExited) {
        Stop-ProcessTree -RootProcessId $Server.Id
    }
}
