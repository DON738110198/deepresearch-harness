param(
    [ValidateRange(3, 20)]
    [int]$Trials = 3,
    [string]$RunLabel = "",
    [string]$Queries = "runs\browsecomp_plus_v0\dev_queries_5.json",
    [ValidateSet("deepseek-v4-flash", "deepseek-v4-pro")]
    [string]$Model = "deepseek-v4-flash",
    [string]$Node = "",
    [string]$EvaluationPython = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TargetManifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\target_manifest.json"
$RetrieverManifest = Join-Path $Root "benchmarks\browsecomp_plus_v0\retriever_candidates.json"
$Partitions = Join-Path $Root "benchmarks\browsecomp_plus_v0\query_partitions.json"
$BM25Script = Join-Path $PSScriptRoot "run_browsecomp_plus_smoke.ps1"
$DenseScript = Join-Path $PSScriptRoot "run_browsecomp_plus_dense_smoke.ps1"
$ControlPolicy = "answer_reserve_nonthinking_v0"

if (-not $EvaluationPython) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $EvaluationPython = $PythonCommand.Source
    }
}

if (-not $RunLabel) {
    $RunLabel = Get-Date -Format "yyyyMMdd-HHmmss"
}
if ($RunLabel -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$") {
    throw "RunLabel must contain only letters, digits, dot, underscore, or hyphen."
}

$RunRootRelative = "runs\browsecomp_plus_v0\repeats\$RunLabel"
$RunRoot = Join-Path $Root $RunRootRelative
if (
    -not $Resume -and
    (Test-Path -LiteralPath $RunRoot) -and
    (Get-ChildItem -LiteralPath $RunRoot)
) {
    throw "Repeat output directory is not empty: $RunRoot"
}
[void](New-Item -ItemType Directory -Path $RunRoot -Force)

foreach ($Path in @(
    $TargetManifest,
    $RetrieverManifest,
    $Partitions,
    $EvaluationPython,
    $BM25Script,
    $DenseScript,
    (Join-Path $Root $Queries)
)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing repeat experiment artifact: $Path"
    }
}
$DependencyCheck = & $EvaluationPython -c `
    "import duckdb, pydantic, deepresearch_harness" 2>&1
$DependencyExitCode = $LASTEXITCODE
if ($DependencyExitCode -ne 0) {
    $DependencyCheck | Out-Host
    throw "EvaluationPython must import duckdb, pydantic, and deepresearch_harness."
}

function Get-NormalizedTextSha256 {
    param([string]$Path)

    $Text = [IO.File]::ReadAllText($Path)
    $Normalized = $Text.Replace("`r`n", "`n").Replace("`r", "`n")
    $Bytes = [Text.Encoding]::UTF8.GetBytes($Normalized)
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $Hash = $Hasher.ComputeHash($Bytes)
    } finally {
        $Hasher.Dispose()
    }
    return -join ($Hash | ForEach-Object { $_.ToString("x2") })
}

function Get-RepositoryRelativePath {
    param([string]$Path)

    return [IO.Path]::GetRelativePath($Root, $Path).Replace("\", "/")
}

function Invoke-Diagnostics {
    param(
        [string]$SourceDirectory,
        [string]$Stem
    )

    $Summary = Join-Path $SourceDirectory "summary.json"
    $Gold = Join-Path $RunRoot "$Stem.gold.json"
    $Diagnostic = Join-Path $RunRoot "$Stem.diagnostic.json"
    $OfficialExport = Join-Path $RunRoot "$Stem.official_input"
    $OfficialExportManifest = Join-Path $OfficialExport "export_manifest.json"
    if (-not (Test-Path -LiteralPath $Gold)) {
        $PrepareOutput = & $EvaluationPython -m deepresearch_harness.cli `
            prepare-browsecomp-plus-dev-gold `
            --manifest $TargetManifest `
            --partitions $Partitions `
            --source-summary $Summary `
            --output $Gold 2>&1
        $PrepareExitCode = $LASTEXITCODE
        $PrepareOutput | Out-Host
        if ($PrepareExitCode -ne 0) {
            throw "Development gold preparation failed for $Stem with exit code $PrepareExitCode"
        }
    } else {
        Write-Host "reuse_gold=$Gold"
    }

    if (-not (Test-Path -LiteralPath $Diagnostic)) {
        $ScoreOutput = & $EvaluationPython -m deepresearch_harness.cli `
            score-browsecomp-plus-diagnostic `
            --source-dir $SourceDirectory `
            --gold-slice $Gold `
            --output $Diagnostic 2>&1
        $ScoreExitCode = $LASTEXITCODE
        $ScoreOutput | Out-Host
        if ($ScoreExitCode -ne 0) {
            throw "Development diagnostic failed for $Stem with exit code $ScoreExitCode"
        }
    } else {
        Write-Host "reuse_diagnostic=$Diagnostic"
    }

    if (-not (Test-Path -LiteralPath $OfficialExportManifest)) {
        if ((Test-Path -LiteralPath $OfficialExport) -and (Get-ChildItem $OfficialExport)) {
            throw "Cannot resume an incomplete official export: $OfficialExport"
        }
        $ExportOutput = & $EvaluationPython -m deepresearch_harness.cli `
            export-pi-browsecomp-runs `
            --source-dir $SourceDirectory `
            --output-dir $OfficialExport 2>&1
        $ExportExitCode = $LASTEXITCODE
        $ExportOutput | Out-Host
        if ($ExportExitCode -ne 0) {
            throw "Official evaluator export failed for $Stem with exit code $ExportExitCode"
        }
    } else {
        Write-Host "reuse_official_export=$OfficialExportManifest"
    }

    return [ordered]@{
        summary_path = Get-RepositoryRelativePath $Summary
        gold_slice_path = Get-RepositoryRelativePath $Gold
        diagnostic_path = Get-RepositoryRelativePath $Diagnostic
        official_export_manifest_path = Get-RepositoryRelativePath $OfficialExportManifest
    }
}

function Invoke-Variant {
    param(
        [int]$Trial,
        [ValidateSet("baseline", "candidate")]
        [string]$Variant
    )

    $TrialName = "trial-{0:d2}" -f $Trial
    $Stem = "$TrialName-$Variant"
    $SourceRelative = "$RunRootRelative\$Stem"
    $SourceDirectory = Join-Path $Root $SourceRelative
    $Summary = Join-Path $SourceDirectory "summary.json"
    if (Test-Path -LiteralPath $Summary) {
        Write-Host "reuse_frozen_summary=$Summary"
    } else {
        if ((Test-Path -LiteralPath $SourceDirectory) -and (Get-ChildItem $SourceDirectory)) {
            throw "Cannot resume an incomplete variant without a frozen summary: $SourceDirectory"
        }
        if ($Variant -eq "baseline") {
            & $BM25Script `
                -Queries $Queries `
                -OutputDir $SourceRelative `
                -Model $Model `
                -ControlPolicy $ControlPolicy `
                -Node $Node | Out-Host
        } else {
            & $DenseScript `
                -Queries $Queries `
                -OutputDir $SourceRelative `
                -Model $Model `
                -ControlPolicy $ControlPolicy `
                -Node $Node | Out-Host
        }
        if (-not (Test-Path -LiteralPath $Summary)) {
            throw "Variant completed without a frozen summary: $SourceDirectory"
        }
    }
    [void](Invoke-Diagnostics -SourceDirectory $SourceDirectory -Stem $Stem)
}

function Get-VariantReference {
    param(
        [int]$Trial,
        [ValidateSet("baseline", "candidate")]
        [string]$Variant
    )

    $Stem = "trial-{0:d2}-$Variant" -f $Trial
    return [ordered]@{
        summary_path = "$($RunRootRelative.Replace('\', '/'))/$Stem/summary.json"
        gold_slice_path = "$($RunRootRelative.Replace('\', '/'))/$Stem.gold.json"
        diagnostic_path = "$($RunRootRelative.Replace('\', '/'))/$Stem.diagnostic.json"
        official_export_manifest_path = "$($RunRootRelative.Replace('\', '/'))/$Stem.official_input/export_manifest.json"
    }
}

$Pairs = @()
for ($Trial = 1; $Trial -le $Trials; $Trial++) {
    $ExecutionOrder = if ($Trial % 2 -eq 1) {
        "baseline_first"
    } else {
        "candidate_first"
    }
    $Pairs += [ordered]@{
        trial_id = "trial-{0:d2}" -f $Trial
        execution_order = $ExecutionOrder
        baseline = Get-VariantReference -Trial $Trial -Variant "baseline"
        candidate = Get-VariantReference -Trial $Trial -Variant "candidate"
    }
}

$ExperimentManifestPath = Join-Path $RunRoot "repeat_experiment.json"
$RegistrationStatus = "pre_generation"
$RegisteredAt = [DateTimeOffset]::UtcNow.ToString("o")
if (Test-Path -LiteralPath $ExperimentManifestPath) {
    if (-not $Resume) {
        throw "Repeat experiment manifest already exists; pass -Resume to validate and continue it."
    }
    $ExistingExperiment = Get-Content -LiteralPath $ExperimentManifestPath -Raw | ConvertFrom-Json
    if ($ExistingExperiment.registration_status -and $ExistingExperiment.registered_at) {
        $RegistrationStatus = $ExistingExperiment.registration_status
        $RegisteredAt = $ExistingExperiment.registered_at
    } else {
        $RegistrationStatus = "reconstructed_after_interruption"
    }
} elseif (Get-ChildItem -LiteralPath $RunRoot -Recurse -Filter "summary.json") {
    $RegistrationStatus = "reconstructed_after_interruption"
}
$Experiment = [ordered]@{
    schema_version = "browsecomp-plus-repeat-experiment-v0"
    registered_at = $RegisteredAt
    registration_status = $RegistrationStatus
    target_manifest_sha256 = Get-NormalizedTextSha256 $TargetManifest
    expected_adapter_version = "pi-browsecomp-v6"
    model = $Model
    control_policy = $ControlPolicy
    baseline_retriever_id = "bm25"
    candidate_retriever_id = "qwen3-embedding-0.6b"
    candidate_retriever_manifest_sha256 = Get-NormalizedTextSha256 $RetrieverManifest
    minimum_trials = $Trials
    pairs = $Pairs
}
$ExperimentJson = $Experiment | ConvertTo-Json -Depth 8
if ($ExistingExperiment) {
    $ExistingCanonical = $ExistingExperiment | ConvertTo-Json -Depth 8 -Compress
    $RequestedCanonical = $Experiment | ConvertTo-Json -Depth 8 -Compress
    if ($ExistingCanonical -ne $RequestedCanonical) {
        throw "Resume arguments do not match the frozen repeat experiment manifest."
    }
} else {
    [IO.File]::WriteAllText(
        $ExperimentManifestPath,
        $ExperimentJson + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

for ($Trial = 1; $Trial -le $Trials; $Trial++) {
    if ($Trial % 2 -eq 1) {
        Invoke-Variant -Trial $Trial -Variant "baseline"
        Invoke-Variant -Trial $Trial -Variant "candidate"
    } else {
        Invoke-Variant -Trial $Trial -Variant "candidate"
        Invoke-Variant -Trial $Trial -Variant "baseline"
    }
}

$ComparisonPath = Join-Path $RunRoot "repeat_comparison.json"
$AggregateArguments = @(
    "-m",
    "deepresearch_harness.cli",
    "aggregate-browsecomp-plus-repeats",
    "--experiment-manifest",
    $ExperimentManifestPath,
    "--target-manifest",
    $TargetManifest,
    "--output",
    $ComparisonPath
)
if ($Resume -and (Test-Path -LiteralPath $ComparisonPath)) {
    $AggregateArguments += "--validate-existing"
}
$AggregateOutput = & $EvaluationPython @AggregateArguments 2>&1
$AggregateExitCode = $LASTEXITCODE
$AggregateOutput | Out-Host
if ($AggregateExitCode -ne 0) {
    throw "Repeat aggregation failed with exit code $AggregateExitCode"
}

Write-Output "repeat_experiment=$ExperimentManifestPath"
Write-Output "repeat_comparison=$ComparisonPath"
