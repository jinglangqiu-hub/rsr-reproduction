param(
    [string]$RunId,
    [switch]$Execute,
    [switch]$AllowMarketTemplate
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
$selected = @($plan.runs | Where-Object { -not $RunId -or $_.id -eq $RunId })
if (-not $selected) { throw "No such run in manifest: $RunId" }
if ($Execute -and -not $RunId) { throw 'Specify a single -RunId before using -Execute.' }

foreach ($run in $selected) {
    $config = $plan.configurations.($run.configuration)
    [pscustomobject]@{
        RunId = $run.id
        Market = $config.market
        Model = $config.method
        Relation = $config.relation
        Seed = $run.seed
        Epochs = $config.epochs
        SequenceLength = $config.seq
        HiddenUnits = $config.unit
        Alpha = $config.alpha
        AdamLearningRate = $config.lr
        Embedding = $config.embedding
        ParameterProvenance = $config.parameter_provenance
    } | Format-List
}

$python = '/home/luckythrower/.venvs/rsr-gpu/bin/python'
$arguments = @()
if ($RunId) { $arguments += @('--run', $RunId) }
if ($Execute) {
    if ($config.parameter_provenance -eq 'market_template' -and -not $AllowMarketTemplate) {
        throw 'This is a market-template run. The author did not publish its exact settings. Use -AllowMarketTemplate only if this comparison has been approved.'
    }
    $answer = Read-Host 'Type START to start the one run shown above'
    if ($answer -cne 'START') { return }
    $arguments += '--execute'
    if ($AllowMarketTemplate) { $arguments += '--allow-market-template' }
    wsl.exe -d Ubuntu --cd $project --exec $python environment/run_gpu.py compat/run_experiment.py @arguments
} else {
    wsl.exe -d Ubuntu --cd $project --exec $python compat/run_experiment.py @arguments
}
exit $LASTEXITCODE
