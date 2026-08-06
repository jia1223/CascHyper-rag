$ErrorActionPreference = 'Stop'

$reviewRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent $reviewRoot
$pythonExecutable = 'C:\Users\kunkun\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$resultRoot = Join-Path $repositoryRoot 'Hyperrag\experiment_results\deepseek-v4-flash\physics'
$runDirectory = Join-Path $reviewRoot 'runs\physics_expert_review_seed_20260806'

& $pythonExecutable (Join-Path $reviewRoot 'prepare_review.py') `
    --casc-results `
      (Join-Path $resultRoot 'hyperrag_v81_1_stage_result.json') `
      (Join-Path $resultRoot 'hyperrag_v81_2_stage_result.json') `
      (Join-Path $resultRoot 'hyperrag_v81_3_stage_result.json') `
    --hyper-results `
      (Join-Path $resultRoot 'hyperrag_main_1_stage_result.json') `
      (Join-Path $resultRoot 'hyperrag_main_2_stage_result.json') `
      (Join-Path $resultRoot 'hyperrag_main_3_stage_result.json') `
    --output-dir $runDirectory `
    --sample-size 60 `
    --calibration-size 5 `
    --expert-count 3 `
    --seed 20260806

if ($LASTEXITCODE -eq 2) {
    Write-Host 'Reference-answer template created. Fill it, then rerun with --references as described in README.md.'
    exit 0
}

exit $LASTEXITCODE
