param(
    [int]$Port = 8765,
    [string]$Queries = "runs\browsecomp_plus_v0\dev_queries_5.json",
    [string]$OutputDir = "runs\browsecomp_plus_v0\pi_flash_smoke",
    [ValidateSet("deepseek-v4-flash", "deepseek-v4-pro")]
    [string]$Model = "deepseek-v4-flash",
    [string]$Node = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Manifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\target_manifest.json"
$Partitions = Join-Path $Root "benchmarks\browsecomp_plus_v0\query_partitions.json"
$Index = Join-Path $Root "runs\_external\browsecomp-plus-indexes-direct\bm25"
$Python = Join-Path $Root "runs\_external\.bcp-runtime-venv\Scripts\python.exe"
$JavaHome = (Resolve-Path (Join-Path $Root "runs\_external\java21\jdk-21.0.12+8")).Path
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
foreach ($Path in @($Manifest, $Partitions, $Index, $Python, $JavaHome, $Adapter, $Node)) {
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
$ServerOut = Join-Path $Root "runs\browsecomp_plus_v0\bm25-server.stdout.log"
$ServerErr = Join-Path $Root "runs\browsecomp_plus_v0\bm25-server.stderr.log"
$Server = Start-Process -FilePath $Python -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $ServerOut -RedirectStandardError $ServerErr `
    -ArgumentList @(
        "-m", "deepresearch_harness.bm25_server",
        "--manifest", $Manifest,
        "--index-path", $Index,
        "--port", $Port
    )

try {
    $Ready = $false
    for ($Attempt = 1; $Attempt -le 120; $Attempt++) {
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
        throw "BM25 server did not become ready. Inspect $ServerErr"
    }

    & python -m deepresearch_harness.cli run-pi-browsecomp-smoke `
        --manifest $Manifest `
        --partitions $Partitions `
        --queries (Join-Path $Root $Queries) `
        --output-dir (Join-Path $Root $OutputDir) `
        --node $Node `
        --adapter-dir $Adapter `
        --search-url "http://127.0.0.1:$Port/search" `
        --model $Model `
        --timeout-seconds 900
    if ($LASTEXITCODE -ne 0) {
        throw "BrowseComp-Plus smoke returned exit code $LASTEXITCODE"
    }
} finally {
    if ($Server -and -not $Server.HasExited) {
        Stop-ProcessTree -RootProcessId $Server.Id
    }
}
