# Deploy this repo's app to the Car-Racing-Agent HF Space.
#
# Usage:
#   # one-time setup
#   git clone https://huggingface.co/spaces/nirmalpratheep/Car-Racing-Agent
#
#   # from the repo root, sync + commit + push
#   powershell -ExecutionPolicy Bypass -File hf_space\deploy.ps1 -SpaceDir ..\Car-Racing-Agent
#
param(
    [Parameter(Mandatory=$true)]
    [string]$SpaceDir,

    [string]$CommitMessage = "Sync from imitationLearning repo"
)

$ErrorActionPreference = "Stop"

$repoRoot  = Resolve-Path (Join-Path $PSScriptRoot "..")
$spaceRoot = Resolve-Path $SpaceDir

Write-Host "Repo:  $repoRoot"
Write-Host "Space: $spaceRoot"
Write-Host ""

# 1) Flatten app.py to root
Copy-Item -Force "$repoRoot\app\app.py"            "$spaceRoot\app.py"

# 2) Support code — full copy of game/ and env/
Copy-Item -Recurse -Force "$repoRoot\game"         "$spaceRoot\game"
Copy-Item -Recurse -Force "$repoRoot\env"          "$spaceRoot\env"

# 3) Training code — only need train_torchrl.py + __init__
New-Item -ItemType Directory -Force -Path "$spaceRoot\training" | Out-Null
Copy-Item -Force "$repoRoot\training\train_torchrl.py" "$spaceRoot\training\train_torchrl.py"
Copy-Item -Force "$repoRoot\training\__init__.py"      "$spaceRoot\training\__init__.py"

# 4) Model checkpoint (~7.5 MB, well under the 10MB LFS threshold)
Copy-Item -Force "$repoRoot\checkpoints\ppo_torchrl_final.pt" "$spaceRoot\ppo_torchrl_final.pt"

# 5) Space config files (README, requirements, .gitignore)
Copy-Item -Force "$PSScriptRoot\README.md"         "$spaceRoot\README.md"
Copy-Item -Force "$PSScriptRoot\requirements.txt"  "$spaceRoot\requirements.txt"
Copy-Item -Force "$PSScriptRoot\.gitignore"        "$spaceRoot\.gitignore"

# 6) Strip bytecode caches that might have tagged along
Get-ChildItem -Path $spaceRoot -Recurse -Force -Include __pycache__ -Directory |
    Remove-Item -Recurse -Force

Write-Host ""
Write-Host "Files synced. Contents of space root:"
Get-ChildItem $spaceRoot | Select-Object Name, Length | Format-Table

Write-Host ""
Write-Host "Next: from $spaceRoot run"
Write-Host "  git add -A"
Write-Host "  git commit -m `"$CommitMessage`""
Write-Host "  git push"
