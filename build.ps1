# Build and deploy the RainWorldRL mod.
#
# Reads game_dir from rainworld_rl.toml (repo root) if present, otherwise uses
# the default Steam path. Passes it to MSBuild as RainWorldDir and copies the
# built DLL into <game_dir>\BepInEx\plugins. Note the DLL is locked while the
# game is running; close Rain World first (or use python -m rainworld_rl.launcher).
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$GameDir = "Z:\SteamLibrary\steamapps\common\Rain World"

$ConfigPath = Join-Path $RepoRoot "rainworld_rl.toml"
if ($env:RAINWORLD_RL_CONFIG) { $ConfigPath = $env:RAINWORLD_RL_CONFIG }
if (Test-Path $ConfigPath) {
    foreach ($line in Get-Content $ConfigPath) {
        if ($line -match '^\s*game_dir\s*=\s*"([^"]+)"') {
            $GameDir = $Matches[1]
            Write-Host "Using game_dir from $ConfigPath" -ForegroundColor DarkGray
            break
        }
    }
}

$GameDirPosix = $GameDir -replace '\\', '/'
$PluginsPath = Join-Path $GameDir "BepInEx\plugins"
$DllPath = Join-Path $RepoRoot "bin\Debug\net472\RainWorldRL.dll"

Write-Host "Building RainWorldRL (RainWorldDir=$GameDirPosix)..." -ForegroundColor Cyan
dotnet build (Join-Path $RepoRoot "RainWorldRL.csproj") -c Debug "-p:RainWorldDir=$GameDirPosix"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build succeeded!" -ForegroundColor Green

if (-not (Test-Path $PluginsPath)) {
    Write-Host "Plugins path not found: $PluginsPath" -ForegroundColor Red
    exit 1
}

Write-Host "Copying to plugins folder..." -ForegroundColor Cyan
Copy-Item $DllPath $PluginsPath -Force

Write-Host "Deployed to: $PluginsPath\RainWorldRL.dll" -ForegroundColor Green
Write-Host "Done! Start Rain World to test." -ForegroundColor Green
