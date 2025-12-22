# Build and deploy RainWorldRL mod
$ErrorActionPreference = "Stop"

$PluginsPath = "Z:\SteamLibrary\steamapps\common\Rain World\BepInEx\plugins"
$DllPath = "bin\Debug\net472\RainWorldRL.dll"

Write-Host "Building RainWorldRL..." -ForegroundColor Cyan
dotnet build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build succeeded!" -ForegroundColor Green

# Check if destination exists
if (-not (Test-Path $PluginsPath)) {
    Write-Host "Plugins path not found: $PluginsPath" -ForegroundColor Red
    exit 1
}

# Copy the DLL
Write-Host "Copying to plugins folder..." -ForegroundColor Cyan
Copy-Item $DllPath $PluginsPath -Force

Write-Host "Deployed to: $PluginsPath\RainWorldRL.dll" -ForegroundColor Green
Write-Host "Done! Start Rain World to test." -ForegroundColor Green

