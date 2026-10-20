# GlucoTwin End-to-End Docker Compose Smoke Test
# Brings up the full stack, verifies health, tests API & streaming, and tears down.

$ErrorActionPreference = "Stop"

Write-Host "======================================================"
Write-Host "Starting GlucoTwin Stack Smoke Test"
Write-Host "======================================================"

$startTime = Get-Date

# 1. Bring up the stack with build
Write-Host "`n[1/6] Building and starting Docker Compose services..."
docker compose up -d --build

# 2. Wait for API to become healthy
Write-Host "`n[2/6] Waiting for services to become healthy..."
$apiHealthy = $false
$maxRetries = 60
$retryCount = 0

while (-not $apiHealthy -and $retryCount -lt $maxRetries) {
    Start-Sleep -Seconds 3
    $retryCount++
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($resp.status -eq "ok" -or $resp.status -eq "healthy") {
            $apiHealthy = $true
            Write-Host "API is healthy! (Attempt $retryCount)"
            break
        }
    } catch {
        Write-Host "Waiting for API... ($retryCount/$maxRetries)"
    }
}

if (-not $apiHealthy) {
    Write-Error "API failed to become healthy within timeout."
    docker compose logs api
    docker compose down
    exit 1
}

$upTime = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 2)
Write-Host "Startup time: $upTime seconds"

# 3. Test REST Endpoints
Write-Host "`n[3/6] Testing REST endpoints..."

# Test /health
Write-Host "Testing GET /health..."
$healthRes = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
Write-Host "Health Response: $($healthRes | ConvertTo-Json -Compress)"

# Test /api/v1/predict
Write-Host "Testing POST /api/v1/predict..."
$predictBody = @{
    patient_id = "DEMO_PAT_001"
    recent_cgm_series = @(142.0, 144.0, 145.0, 148.0, 150.0)
    recent_carbs_g = 35.0
} | ConvertTo-Json

$predictRes = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $predictBody -ContentType "application/json"
Write-Host "Predict Response: Patient=$($predictRes.patient_id), Current=$($predictRes.current_glucose), Horizons=$($predictRes.forecast_horizons.Count)"

# Test /api/v1/simulate
Write-Host "Testing POST /api/v1/simulate..."
$simBody = @{
    patient_id = "DEMO_PAT_001"
    interventions = @(
        @{
            intervention_type = "meal"
            carbs_g = 45.0
        }
    )
} | ConvertTo-Json

$simRes = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/simulate" -Method Post -Body $simBody -ContentType "application/json"
Write-Host "Simulate Response: Peak Change=$($simRes.peak_change_mgdl) mg/dL, Constraint Applied=$($simRes.constraint_applied)"

# Test /api/v1/summary/{id}
Write-Host "Testing GET /api/v1/summary/DEMO_PAT_001..."
$sumRes = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/summary/DEMO_PAT_001" -Method Get
Write-Host "Summary Response: Patient=$($sumRes.patient_name), TIR=$($sumRes.metrics.time_in_range_percentage)%"

# 4. Test WebSocket stream at 60x replay speed
Write-Host "`n[4/6] Testing WebSocket stream at 60x speed (requiring >= 5 ticks, measuring latency)..."
.venv\Scripts\python.exe scripts/test_ws_stream.py DEMO_PAT_001 45 5 20
if ($LASTEXITCODE -ne 0) {
    Write-Error "WebSocket stream test failed."
    docker compose down -v
    exit 1
}


# 5. Report Container Image Sizes
Write-Host "`n[5/6] Measuring container image sizes..."
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | Select-String -Pattern "glucotwin|timescale|apache/kafka|mlflow|prometheus|grafana"

# 6. Show Compose Service Status
Write-Host "`nDocker Compose Status:"
docker compose ps

# 7. Teardown stack
Write-Host "`n[6/6] Tearing down Docker Compose stack..."
docker compose down -v
Write-Host "Teardown complete."


$totalDuration = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 2)
Write-Host "`n======================================================"
Write-Host "Smoke Test PASSED in $totalDuration seconds!"
Write-Host "======================================================"
