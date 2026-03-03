param(
  [string]$OutputDir = ".",
  [string]$ArchiveName = "unitree_g1_mujoco_ppo_hpc_bundle.zip"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$outDirAbs = Resolve-Path $OutputDir
$archivePath = Join-Path $outDirAbs $ArchiveName

if (Test-Path $archivePath) {
  Remove-Item -Force $archivePath
}

$items = @(
  "README.md",
  ".gitignore",
  "scripts",
  "slurm"
)

$paths = @()
foreach ($item in $items) {
  $p = Join-Path $root $item
  if (Test-Path $p) {
    $paths += $p
  }
}

if ($paths.Count -eq 0) {
  throw "No files found to package in $root"
}

Compress-Archive -Path $paths -DestinationPath $archivePath -Force
Write-Host "Created archive: $archivePath"
