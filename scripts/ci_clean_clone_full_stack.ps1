# Clean clone full stack reproduction check
# Clones this repo into a temp folder (committed state plus staged/modified files,
# excluding data/raw, data/processed, and data/demo).
# Runs docker compose up --build -d in DEMO_MODE.
# Verifies container health with timeouts.
# Checks API endpoints via curl.exe or Invoke-RestMethod.
# Runs scripts/capture_screenshots.mjs against that stack.
# Tears down the stack.

$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host "Starting Clean Clone Full Stack Reproduction"
Write-Host "=========================================================="

$repoRoot = (Get-Location).Path
$tempBase = [System.IO.Path]::GetTempPath()
$cloneDir = Join-Path $tempBase ("glucotwin_stack_clone_" + [System.Guid]::NewGuid().ToString().Substring(0, 8))

Write-Host "`n[1/6] Cloning repository into $cloneDir..."
git clone $repoRoot $cloneDir

# Copy staged files and modified working tree files into the clone
Write-Host "Applying staged and modified working tree files to clone..."
$filesToCopy = (git diff --name-only --cached) + (git diff --name-only) + (git ls-files --others --exclude-standard)
$filesToCopy = $filesToCopy | Sort-Object -Unique

foreach ($f in $filesToCopy) {
    if ([string]::IsNullOrWhiteSpace($f)) { continue }
    # Skip any clinical data files
    if ($f -match '^[Dd]ata[/\\](raw|processed|demo)') { continue }

    $srcPath = Join-Path $repoRoot $f
    $dstPath = Join-Path $cloneDir $f
    $dstDir = Split-Path $dstPath -Parent

    if (-not (Test-Path -LiteralPath $srcPath)) {
        if (Test-Path -LiteralPath $dstPath) {
            Remove-Item -LiteralPath $dstPath -Force -ErrorAction SilentlyContinue
        }
        continue
    }
    if (-not (Test-Path -LiteralPath $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $srcPath -Destination $dstPath -Force
}

# Verify no raw, processed, or demo clinical data files exist in the clone
Write-Host "`n[2/6] Verifying no raw, processed, or demo parquet files exist in clean clone..."
$clinicalDataDirs = @("data/raw", "data/processed", "data/demo")
$dataFiles = @()
foreach ($d in $clinicalDataDirs) {
    $targetPath = Join-Path $cloneDir $d
    if (Test-Path $targetPath) {
        $files = Get-ChildItem -Path $targetPath -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".gitkeep" }
        $dataFiles += $files
    }
}
if ($dataFiles.Count -gt 0) {
    Write-Warning "Removing unexpected clinical data files found in clean clone..."
    $dataFiles | Remove-Item -Force
}
$remainingDataFiles = @()
foreach ($d in $clinicalDataDirs) {
    $targetPath = Join-Path $cloneDir $d
    if (Test-Path $targetPath) {
        $files = Get-ChildItem -Path $targetPath -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".gitkeep" }
        $remainingDataFiles += $files
    }
}
Write-Host "Clean clone clinical data files count: $($remainingDataFiles.Count) (expected: 0)"
if ($remainingDataFiles.Count -ne 0) {
    Write-Error "Clean clone contains unexpected data files!"
    exit 1
}

# Tear down any running stack in repo root to free host ports
Write-Host "`n[3/6] Ensuring host ports are free by stopping any existing repo containers..."
docker compose down --remove-orphans

Push-Location $cloneDir
try {
    # Set DEMO_MODE=true
    $env:DEMO_MODE = "true"

    Write-Host "`n[4/6] Building and starting full Docker Compose stack in DEMO_MODE..."
    cmd.exe /c "docker compose up --build -d"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker compose up --build -d failed"
        exit 1
    }

    Write-Host "Waiting for services to become healthy (timeout: 180s)..."
    $maxWaitSec = 180
    $startTime = Get-Date
    $allHealthy = $false

    while (((Get-Date) - $startTime).TotalSeconds -lt $maxWaitSec) {
        Start-Sleep -Seconds 4

        # Check frontend on port 3000
        $frontendOk = $false
        try {
            $fRes = curl.exe --max-time 3 -s -o NUL -w "%{http_code}" http://127.0.0.1:3000
            if ($fRes -eq "200") { $frontendOk = $true }
        } catch {}

        # Check backend on port 8000
        $backendOk = $false
        try {
            $bRes = curl.exe --max-time 3 -s -o NUL -w "%{http_code}" http://127.0.0.1:8000/health
            if ($bRes -eq "200") { $backendOk = $true }
        } catch {}

        # Check docker compose ps statuses
        $psOutput = cmd.exe /c "docker compose ps" | Out-String
        $apiHealthy = $psOutput -match "glucotwin-api.*\(healthy\)"
        $feHealthy = $psOutput -match "glucotwin-frontend.*\(healthy\)"

        Write-Host "  Health check status: Backend=$backendOk (healthy=$apiHealthy), Frontend=$frontendOk (healthy=$feHealthy)"

        if ($backendOk -and $frontendOk -and $apiHealthy -and $feHealthy) {
            $allHealthy = $true
            break
        }
    }

    if (-not $allHealthy) {
        Write-Error "Timeout waiting for all Docker Compose services to become healthy."
        cmd.exe /c "docker compose ps"
        exit 1
    }

    Write-Host "PASS: Full stack is UP and all services are healthy!"
    cmd.exe /c "docker compose ps"

    # [5/6] Query API endpoints using curl.exe / Invoke-RestMethod
    Write-Host "`n[5/6] Testing API endpoints against clean clone stack in DEMO_MODE..."

    # Check 1: /api/v1/twin/patients lists 3 demo patients and 4 personas
    Write-Host "Querying /api/v1/twin/patients..."
    $patients = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/twin/patients" -TimeoutSec 10
    $patientIds = $patients | ForEach-Object { if ($_.id) { $_.id } else { $_.patient_id } }
    Write-Host "Found $($patientIds.Count) total patients in registry: $($patientIds -join ', ')"

    $demoExpected = @("DEMO_PAT_001", "DEMO_PAT_002", "DEMO_PAT_003")
    $personaExpected = @("IND_PAT_001", "IND_PAT_002", "IND_PAT_003", "IND_PAT_004")

    foreach ($d in $demoExpected) {
        if ($patientIds -notcontains $d) {
            Write-Error "Missing expected demo patient: $d"
            exit 1
        }
    }
    Write-Host "  PASS: 3 demo patients present (DEMO_PAT_001, DEMO_PAT_002, DEMO_PAT_003)"

    foreach ($p in $personaExpected) {
        if ($patientIds -notcontains $p) {
            Write-Error "Missing expected synthetic persona: $p"
            exit 1
        }
    }
    Write-Host "  PASS: 4 synthetic personas present (IND_PAT_001 through IND_PAT_004)"

    # Check 2: /api/v1/twin/DEMO_PAT_001/cgm returns synthetic series
    Write-Host "`nQuerying /api/v1/twin/DEMO_PAT_001/cgm?days=1..."
    $cgm = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/twin/DEMO_PAT_001/cgm?days=1" -TimeoutSec 10
    Write-Host "Received $($cgm.Count) CGM readings for DEMO_PAT_001"
    if ($cgm.Count -eq 0) {
        Write-Error "Expected synthetic CGM readings for DEMO_PAT_001, but got 0!"
        exit 1
    }
    $firstReading = $cgm[0]
    Write-Host "  First reading: timestamp=$($firstReading.timestamp), glucose_mgdl=$($firstReading.glucose_mgdl)"
    Write-Host "  PASS: Synthetic CGM series generated automatically on startup in DEMO_MODE without pre-existing parquet files."

    # Check 3: /api/v1/predict answers for DEMO_PAT_001
    Write-Host "`nQuerying /api/v1/predict for DEMO_PAT_001..."
    $recentGlucose = @(160.0, 162.0, 164.5, 167.0, 171.0)
    $predictPayload = @{
        patient_id = "DEMO_PAT_001"
        recent_cgm_series = $recentGlucose
    } | ConvertTo-Json

    $pred = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $predictPayload -ContentType "application/json" -TimeoutSec 10
    Write-Host "Predict response: patient_id=$($pred.patient_id), warning=$($pred.adverse_event_warning), horizons=$($pred.forecast_horizons.Count)"
    if ($pred.forecast_horizons.Count -eq 0) {
        Write-Error "Predict endpoint returned empty forecast_horizons!"
        exit 1
    }
    Write-Host "  PASS: /api/v1/predict answered successfully for DEMO_PAT_001."

    # [6/6] Run scripts/capture_screenshots.mjs against this clean clone stack
    Write-Host "`n[6/6] Running scripts/capture_screenshots.mjs against clean clone stack..."
    # Junction frontend/node_modules from repo root to run playwright
    $cloneNodeModules = Join-Path $cloneDir "frontend\node_modules"
    $rootNodeModules = Join-Path $repoRoot "frontend\node_modules"
    if (-not (Test-Path $cloneNodeModules)) {
        cmd.exe /c "mklink /J `"$cloneNodeModules`" `"$rootNodeModules`"" | Out-Null
    }

    node scripts/capture_screenshots.mjs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "capture_screenshots.mjs failed on clean clone stack"
        exit 1
    }

    # Copy generated screenshots back to repoRoot
    $srcScreenshots = Join-Path $cloneDir "docs\screenshots"
    $dstScreenshots = Join-Path $repoRoot "docs\screenshots"
    Copy-Item -Path (Join-Path $srcScreenshots "*.png") -Destination $dstScreenshots -Force
    Write-Host "PASS: Screenshots regenerated and copied to docs/screenshots/"

    Write-Host "`n=========================================================="
    Write-Host "Clean Clone Full Stack Check: ALL PASSED"
    Write-Host "=========================================================="

} finally {
    Write-Host "`nCleaning up Docker stack and temporary directory..."
    cmd.exe /c "docker compose down -v"
    Pop-Location

    # Remove node_modules junction if present before deleting clone directory
    $cloneNodeModules = Join-Path $cloneDir "frontend\node_modules"
    if (Test-Path $cloneNodeModules) {
        cmd.exe /c "rmdir `"$cloneNodeModules`"" | Out-Null
    }
    Remove-Item -Path $cloneDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Clean up complete."
}
