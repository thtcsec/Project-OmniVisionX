# Demo morning check — NO docker build. Requires stack already running.
# Usage: pwsh -File scripts/demo-healthcheck.ps1

$ErrorActionPreference = "Continue"
Write-Host "`n=== Docker containers (expect omni-*) ===" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>&1

$checks = @(
    @{ Name = "API /health"; Url = "http://127.0.0.1:8080/health" }
    @{ Name = "API integrations status"; Url = "http://127.0.0.1:8080/api/integrations/status" }
    @{ Name = "Agora status"; Url = "http://127.0.0.1:8080/api/agora/status" }
)

Write-Host "`n=== HTTP checks (API on :8080) ===" -ForegroundColor Cyan
foreach ($c in $checks) {
    try {
        $r = Invoke-WebRequest -Uri $c.Url -UseBasicParsing -TimeoutSec 5
        Write-Host ("[OK]  {0} -> {1}" -f $c.Name, $r.StatusCode) -ForegroundColor Green
    }
    catch {
        Write-Host ("[FAIL] {0} — {1}" -f $c.Name, $_.Exception.Message) -ForegroundColor Yellow
    }
}

Write-Host "`n=== Optional (GPU workers, if started) ===" -ForegroundColor Cyan
try {
    $o = Invoke-WebRequest -Uri "http://127.0.0.1:8555/health" -UseBasicParsing -TimeoutSec 3
    Write-Host ("[OK]  omni-object :8555 -> {0}" -f $o.StatusCode) -ForegroundColor Green
}
catch {
    Write-Host "[SKIP] omni-object :8555 (not running or no GPU profile)" -ForegroundColor DarkGray
}

Write-Host "`nDone. For vision demo you need: --profile gpu + NVIDIA + weights under data/weights." -ForegroundColor Cyan
