param(
    [int]$Port = 8769,
    [string]$ServiceDir = "runs\services\obligation-span-opening-v0-20260816-restart1"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = (Resolve-Path (Join-Path $Root ".venv\Scripts\python.exe")).Path
$JavaHome = (Resolve-Path (Join-Path $Root "runs\_external\java21\jdk-21.0.12+8")).Path
$ServiceRoot = Join-Path $Root $ServiceDir
$ExpectedRetrieverId = "bm25-anchor-5+qwen3-embedding-0.6b-lead-15-query_window_v0-64-anchor-reopen-answer_obligation_window_v0"

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port already has a listener."
}
New-Item -ItemType Directory -Path $ServiceRoot -Force | Out-Null
$env:JAVA_HOME = $JavaHome
$env:Path = "$JavaHome\bin;$env:Path"

$Arguments = @(
    "-m", "deepresearch_harness.progressive_disclosure_server",
    "--manifest", "benchmarks/browsecomp_plus_v0/target_manifest.json",
    "--retriever-manifest", "benchmarks/browsecomp_plus_v0/retriever_candidates.json",
    "--candidate-id", "qwen3-embedding-0.6b",
    "--model-dir", "runs/_external/models/Qwen3-Embedding-0.6B",
    "--index-root", "runs/_external/browsecomp-plus-indexes-direct",
    "--document-index-path", "runs/_external/browsecomp-plus-indexes-direct/bm25",
    "--snippet-tokenizer-dir", "C:/Users/73811/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "--anchor-count", "5",
    "--dense-lead-count", "15",
    "--dense-pool-size", "20",
    "--anchor-token-cap", "512",
    "--lead-token-cap", "64",
    "--lead-preview-policy", "query_window_v0",
    "--open-token-cap", "512",
    "--anchor-open-policy", "reopen_with_obligation",
    "--open-content-policy", "answer_obligation_window_v0",
    "--maximum-open-calls", "8",
    "--total-evidence-ingress-token-budget", "33280",
    "--open-evidence-ingress-token-budget", "4096",
    "--port", "$Port"
)
$Stdout = Join-Path $ServiceRoot "stdout.log"
$Stderr = Join-Path $ServiceRoot "stderr.log"
$StartedAt = Get-Date
$Process = Start-Process -FilePath $Python -WindowStyle Hidden -PassThru `
    -WorkingDirectory $Root -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr

try {
    $Health = $null
    for ($Attempt = 1; $Attempt -le 180; $Attempt++) {
        if ($Process.HasExited) {
            throw "Candidate service exited with code $($Process.ExitCode). Inspect $Stderr"
        }
        try {
            $Candidate = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($Candidate.status -eq "ok") {
                $Health = $Candidate
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if ($null -eq $Health) {
        throw "Candidate service did not become healthy. Inspect $Stderr"
    }
    if ($Health.retriever_id -ne $ExpectedRetrieverId) {
        throw "Unexpected retriever id: $($Health.retriever_id)"
    }
    $Listener = Get-NetTCPConnection -LocalAddress "127.0.0.1" `
        -LocalPort $Port -State Listen | Select-Object -First 1
    $Record = [ordered]@{
        schema_version = "local-service-process-v0"
        service_id = "obligation-span-opening-v0"
        status = "healthy"
        launcher_pid = $Process.Id
        listener_pid = $Listener.OwningProcess
        started_at = $StartedAt.ToString("o")
        health_checked_at = (Get-Date).ToString("o")
        url = "http://127.0.0.1:$Port/search"
        retriever_id = $Health.retriever_id
        stdout = $Stdout
        stderr = $Stderr
        command = @($Python) + $Arguments
    }
    $Record | ConvertTo-Json -Depth 5 | Set-Content `
        -LiteralPath (Join-Path $ServiceRoot "process.json") -Encoding utf8
    $Record | ConvertTo-Json -Depth 5
} catch {
    if ($Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
    throw
}
