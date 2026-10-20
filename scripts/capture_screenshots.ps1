# Capture GlucoTwin Dashboard Screenshots using Playwright
$ErrorActionPreference = "Stop"
node scripts/capture_screenshots.mjs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Screenshot capture failed."
    exit 1
}
Write-Host "Screenshots captured successfully in docs/screenshots."
