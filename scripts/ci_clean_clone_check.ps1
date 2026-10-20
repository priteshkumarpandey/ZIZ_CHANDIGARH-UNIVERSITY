# Clean clone CI reproduction check
# Clones this repository from its local path into a temporary folder,
# applies currently staged files AND working tree modifications using git diff / copy, and runs:
# 1. pytest (with no raw or processed data present)
# 2. npm ci && npm run build in frontend/
# 3. docker build for frontend and api images

$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host "Starting Clean Clone CI Verification"
Write-Host "=========================================================="

$repoRoot = (Get-Location).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment python not found at $venvPython"
    exit 1
}

$tempBase = [System.IO.Path]::GetTempPath()
$cloneDir = Join-Path $tempBase ("glucotwin_ci_clone_" + [System.Guid]::NewGuid().ToString().Substring(0, 8))

Write-Host "`n[1/5] Cloning repository into $cloneDir..."
git clone $repoRoot $cloneDir

# Copy/apply staged files AND modified working tree files into the clone
Write-Host "Exporting staged and modified files to clone..."
$filesToCopy = (git diff --name-only --cached) + (git diff --name-only) + (git ls-files --others --exclude-standard)
$filesToCopy = $filesToCopy | Sort-Object -Unique

foreach ($f in $filesToCopy) {
    if ([string]::IsNullOrWhiteSpace($f)) { continue }
    $srcPath = Join-Path $repoRoot $f
    $dstPath = Join-Path $cloneDir $f
    $dstDir = Split-Path $dstPath -Parent
    if (-not (Test-Path -LiteralPath $srcPath)) {
        if (Test-Path -LiteralPath $dstPath) {
            Remove-Item -LiteralPath $dstPath -Force -ErrorAction SilentlyContinue
            Write-Host "  Removed file: $f"
        }
        continue
    }
    if (-not (Test-Path -LiteralPath $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $srcPath -Destination $dstPath -Force
    Write-Host "  Applied file: $f"
}

Push-Location $cloneDir
try {
    # Verify no raw data or processed data exists in clean clone
    $rawFiles = Get-ChildItem -Path "data/raw" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".gitkeep" }
    $procFiles = Get-ChildItem -Path "data/processed" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".gitkeep" }
    Write-Host "`nClean clone data check: raw files=$($rawFiles.Count), processed files=$($procFiles.Count)"

    # 1. Pytest
    Write-Host "`n[2/5] Running pytest in clean clone (no clinical data)..."
    & $venvPython -m pytest -v -m "not requires_data"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pytest failed on clean clone"
        exit 1
    }
    Write-Host "PASS: pytest without data completed successfully."

    # Also run full pytest to ensure requires_data tests skip gracefully
    Write-Host "`nRunning full pytest suite to verify clean skips..."
    & $venvPython -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pytest with graceful skips failed on clean clone"
        exit 1
    }
    Write-Host "PASS: all tests passed or skipped cleanly."

    # 2. Frontend npm ci, checks, & build
    Write-Host "`n[3/5] Running frontend npm ci, lint, type-check, test, and build..."
    Push-Location "frontend"
    try {
        $env:CI = "true"
        cmd.exe /c "npm ci"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "npm ci failed"
            exit 1
        }
        cmd.exe /c "npm run lint:style"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "npm run lint:style failed"
            exit 1
        }
        cmd.exe /c "npm run type-check"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "npm run type-check failed"
            exit 1
        }
        cmd.exe /c "npm run lint"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "npm run lint failed"
            exit 1
        }
        cmd.exe /c "npm test"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "npm test failed"
            exit 1
        }
        cmd.exe /c "npm run build"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "npm run build failed"
            exit 1
        }
        Write-Host "PASS: frontend checks and build succeeded on clean clone."
    } finally {
        Pop-Location
    }

    # 3. Docker build frontend
    Write-Host "`n[4/5] Building Docker frontend image from clean clone..."
    $dockerFrontendSuccess = $false
    for ($i = 1; $i -le 3; $i++) {
        cmd.exe /c "docker build -f deploy/docker/Dockerfile.frontend -t glucotwin-frontend-clean:test ."
        if ($LASTEXITCODE -eq 0) {
            $dockerFrontendSuccess = $true
            break
        }
        Write-Warning "Docker build frontend attempt $i failed. Retrying in 3 seconds..."
        Start-Sleep -Seconds 3
    }
    if (-not $dockerFrontendSuccess) {
        Write-Error "docker build frontend failed"
        exit 1
    }
    Write-Host "PASS: Docker frontend image built successfully."

    # 4. Docker build Backend
    Write-Host "`n[5/5] Building Docker Backend image from clean clone..."
    $dockerBackendSuccess = $false
    for ($i = 1; $i -le 3; $i++) {
        cmd.exe /c "docker build -f deploy/docker/Dockerfile.backend -t glucotwin-backend-clean:test ."
        if ($LASTEXITCODE -eq 0) {
            $dockerBackendSuccess = $true
            break
        }
        Write-Warning "Docker build backend attempt $i failed. Retrying in 3 seconds..."
        Start-Sleep -Seconds 3
    }
    if (-not $dockerBackendSuccess) {
        Write-Error "docker build backend failed"
        exit 1
    }
    Write-Host "PASS: Docker Backend image built successfully."

    Write-Host "`n=========================================================="
    Write-Host "Clean Clone CI Check: ALL PASSED"
    Write-Host "=========================================================="
} finally {
    Pop-Location
    Write-Host "Cleaning up temporary clone directory $cloneDir..."
    Remove-Item -Path $cloneDir -Recurse -Force -ErrorAction SilentlyContinue
}
