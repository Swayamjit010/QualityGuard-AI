# Run the QualityGuard AI server on http://127.0.0.1:8000 (used by the "QualityGuard AI" scheduled task at logon).
# The public address https://qualityguard.tail945149.ts.net is provided by Tailscale Funnel, which proxies to this port
# (configured once with: tailscale funnel --bg 8000). Restarts the server automatically if it exits.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "artifacts\logs" | Out-Null
while ($true) {
  # --proxy-headers: rate limits use each visitor's real IP forwarded by the local proxy, not 127.0.0.1
  python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --proxy-headers `
    --forwarded-allow-ips 127.0.0.1 *>> "artifacts\logs\app_server.log"
  Add-Content "artifacts\logs\app_server.log" "$(Get-Date -Format s) server exited (code $LASTEXITCODE); restarting in 10 s"
  Start-Sleep 10
}
