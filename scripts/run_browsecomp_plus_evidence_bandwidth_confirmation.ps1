param(
    [double]$MaximumCombinedProviderCostUsd = 2.5,
    [int]$MaximumResumeInvocations = 3
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$GoldPython = "G:\DevelopTools\Miniforge\python.exe"
$Node = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
$AdapterDir = Join-Path $Root "integrations\pi-browsecomp"
$AdapterRunner = Join-Path $AdapterDir "v8\runner.mjs"
$Manifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\target_manifest.json"
$Partitions = Join-Path $Root "benchmarks\browsecomp_plus_v0\query_partitions.json"
$Queries = Join-Path $Root "runs\browsecomp_plus_v0\evidence_bandwidth_v0_20260814\dev_queries.json"
$CandidateManifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\evidence_bandwidth_candidate_v0.json"
$RunRoot = Join-Path $Root "runs\browsecomp_plus_v0\repeats\evidence-bandwidth-confirmation-v0-fresh25-20260814"
$ProgressPath = Join-Path $RunRoot "execution_progress.json"
$env:PYTHONPATH = Join-Path $Root "src"

if (-not $env:DEEPSEEK_API_KEY) {
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
        "DEEPSEEK_API_KEY", "User"
    )
}
if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is not set in the process or user environment."
}
foreach ($Path in @(
    $Python,
    $GoldPython,
    $Node,
    $AdapterRunner,
    $Manifest,
    $Partitions,
    $Queries,
    $CandidateManifest,
    (Join-Path $RunRoot "repeat_experiment.json")
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing frozen runtime artifact: $Path"
    }
}
foreach ($Url in @("http://127.0.0.1:8765/health", "http://127.0.0.1:8767/health")) {
    $Health = Invoke-RestMethod -Uri $Url -TimeoutSec 10
    if ($Health.status -ne "ok") {
        throw "Search server is not healthy: $Url"
    }
}

function Get-CombinedCost {
    $Total = 0.0
    Get-ChildItem -LiteralPath $RunRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object {
            $SummaryPath = Join-Path $_.FullName "summary.json"
            if (Test-Path -LiteralPath $SummaryPath) {
                $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
                $Total += [double]$Summary.total_cost_usd
            }
        }
    return $Total
}

function Write-ExecutionProgress {
    param([string]$LatestVariant)

    $Rows = @()
    Get-ChildItem -LiteralPath $RunRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name |
        ForEach-Object {
            $SummaryPath = Join-Path $_.FullName "summary.json"
            if (Test-Path -LiteralPath $SummaryPath) {
                $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
                $Rows += [ordered]@{
                    variant = $_.Name
                    succeeded = $Summary.succeeded
                    failed = $Summary.failed
                    budget_exhausted = $Summary.budget_exhausted
                    attempts = ($Summary.items.attempt_count | Measure-Object -Sum).Sum
                    total_tokens = $Summary.total_tokens
                    cost_usd = $Summary.total_cost_usd
                    summary_sha256 = (Get-FileHash -LiteralPath $SummaryPath -Algorithm SHA256).Hash.ToLower()
                }
            }
        }
    $Payload = [ordered]@{
        schema_version = "browsecomp-plus-evidence-bandwidth-execution-progress-v0"
        updated_at = (Get-Date -Format o)
        status = "generation_in_progress"
        latest_completed_variant = $LatestVariant
        completed_variants = $Rows.Count
        combined_provider_cost_usd = Get-CombinedCost
        maximum_combined_provider_cost_usd = $MaximumCombinedProviderCostUsd
        variants = $Rows
    }
    $Temporary = "$ProgressPath.tmp"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Temporary -Encoding utf8
    Move-Item -LiteralPath $Temporary -Destination $ProgressPath -Force
}

function Invoke-PiVariant {
    param(
        [string]$Name,
        [string]$SearchUrl,
        [string]$RetrieverId,
        [int]$MaxSearchResults,
        [bool]$UseCandidateManifest
    )

    $OutputDir = Join-Path $RunRoot $Name
    $LogPath = Join-Path $RunRoot "$Name.generation.log"
    $ResumeCount = 0
    while ($true) {
        $SummaryPath = Join-Path $OutputDir "summary.json"
        if (Test-Path -LiteralPath $SummaryPath) {
            $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
            if (
                $Summary.succeeded -eq $Summary.query_count -and
                $Summary.failed -eq 0 -and
                $Summary.budget_exhausted -eq 0
            ) {
                break
            }
            if ($Summary.failed -eq 0) {
                throw "$Name has immutable budget-exhausted queries; refusing to retry."
            }
            if ($ResumeCount -ge $MaximumResumeInvocations) {
                throw "$Name exceeded the preregistered failed-only resume limit."
            }
            $ResumeCount += 1
            $Resume = $true
        } else {
            $Resume = $false
        }

        if ((Get-CombinedCost) -ge $MaximumCombinedProviderCostUsd) {
            throw "Combined provider cost reached the preregistered ceiling."
        }
        $Arguments = @(
            "-m", "deepresearch_harness.cli", "run-pi-browsecomp-smoke",
            "--manifest", $Manifest,
            "--partitions", $Partitions,
            "--queries", $Queries,
            "--output-dir", $OutputDir,
            "--node", $Node,
            "--adapter-dir", $AdapterDir,
            "--adapter-runner", $AdapterRunner,
            "--search-url", $SearchUrl,
            "--model", "deepseek-v4-flash",
            "--control-policy", "answer_reserve_nonthinking_v0",
            "--retriever-id", $RetrieverId,
            "--max-search-results", $MaxSearchResults,
            "--timeout-seconds", 900
        )
        if ($UseCandidateManifest) {
            $Arguments += @("--retriever-manifest", $CandidateManifest)
        }
        if ($Resume) {
            $Arguments += "--resume-failed"
        }
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        if ($LASTEXITCODE -notin @(0, 1)) {
            throw "$Name runner returned unexpected exit code $LASTEXITCODE"
        }
    }

    $GoldPath = Join-Path $RunRoot "$Name.gold.json"
    if (-not (Test-Path -LiteralPath $GoldPath)) {
        & $GoldPython -m deepresearch_harness.cli prepare-browsecomp-plus-dev-gold `
            --manifest $Manifest `
            --partitions $Partitions `
            --source-summary (Join-Path $OutputDir "summary.json") `
            --output $GoldPath
        if ($LASTEXITCODE -ne 0) {
            throw "$Name gold projection failed with exit code $LASTEXITCODE"
        }
    }
    $DiagnosticPath = Join-Path $RunRoot "$Name.diagnostic.json"
    if (-not (Test-Path -LiteralPath $DiagnosticPath)) {
        & $Python -m deepresearch_harness.cli score-browsecomp-plus-diagnostic `
            --source-dir $OutputDir `
            --gold-slice $GoldPath `
            --output $DiagnosticPath
        if ($LASTEXITCODE -ne 0) {
            throw "$Name diagnostic failed with exit code $LASTEXITCODE"
        }
    }
    $OfficialDir = Join-Path $RunRoot "$Name.official_input"
    if (-not (Test-Path -LiteralPath (Join-Path $OfficialDir "export_manifest.json"))) {
        & $Python -m deepresearch_harness.cli export-pi-browsecomp-runs `
            --source-dir $OutputDir `
            --output-dir $OfficialDir
        if ($LASTEXITCODE -ne 0) {
            throw "$Name official export failed with exit code $LASTEXITCODE"
        }
    }
    Write-ExecutionProgress -LatestVariant $Name
}

$Execution = @(
    @{ Name = "trial-01-baseline"; Url = "http://127.0.0.1:8765/search"; Retriever = "bm25"; Max = 5; Candidate = $false },
    @{ Name = "trial-01-candidate"; Url = "http://127.0.0.1:8767/search"; Retriever = "qwen3-embedding-0.6b-ebx20-1792"; Max = 20; Candidate = $true },
    @{ Name = "trial-02-candidate"; Url = "http://127.0.0.1:8767/search"; Retriever = "qwen3-embedding-0.6b-ebx20-1792"; Max = 20; Candidate = $true },
    @{ Name = "trial-02-baseline"; Url = "http://127.0.0.1:8765/search"; Retriever = "bm25"; Max = 5; Candidate = $false },
    @{ Name = "trial-03-baseline"; Url = "http://127.0.0.1:8765/search"; Retriever = "bm25"; Max = 5; Candidate = $false },
    @{ Name = "trial-03-candidate"; Url = "http://127.0.0.1:8767/search"; Retriever = "qwen3-embedding-0.6b-ebx20-1792"; Max = 20; Candidate = $true }
)

foreach ($Variant in $Execution) {
    Invoke-PiVariant `
        -Name $Variant.Name `
        -SearchUrl $Variant.Url `
        -RetrieverId $Variant.Retriever `
        -MaxSearchResults $Variant.Max `
        -UseCandidateManifest $Variant.Candidate
}

$Progress = Get-Content -LiteralPath $ProgressPath -Raw | ConvertFrom-Json
$Progress.status = "generation_complete"
$Progress.updated_at = Get-Date -Format o
$Progress | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ProgressPath -Encoding utf8
Write-Output "generation_complete=true"
Write-Output "completed_variants=$($Progress.completed_variants)"
Write-Output "combined_provider_cost_usd=$($Progress.combined_provider_cost_usd)"
