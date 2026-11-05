# Start QualityGuard AI and a public Cloudflare quick tunnel (no Cloudflare account needed).
#   powershell -ExecutionPolicy Bypass -File scripts\start_public.ps1
# The public https://<random>.trycloudflare.com URL is printed below. It changes every time this script starts and works
# only while this window stays open and the PC is on. Close the window (or press Ctrl+C) to take the app offline.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "artifacts\logs" | Out-Null
$cf = @("C:\Program Files (x86)\cloudflared\cloudflared.exe", "C:\Program Files\cloudflared\cloudflared.exe") |
      Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $cf) { throw "cloudflared not found - install it: winget install Cloudflare.cloudflared" }
if (-not (Test-Path "frontend\dist\index.html")) { throw "frontend not built - run: cd frontend; npm install; npm run build" }

# --proxy-headers: rate limits use each visitor's real IP (forwarded by the tunnel), not 127.0.0.1
$app = Start-Process python -ArgumentList "-m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips 127.0.0.1" `
       -RedirectStandardOutput "artifacts\logs\app_server.log" -RedirectStandardError "artifacts\logs\app_server.err.log" -PassThru -NoNewWindow
for ($i = 0; $i -lt 90; $i++) {
  try { if ((Invoke-RestMethod http://127.0.0.1:8000/api/health -TimeoutSec 2).status -eq "ok") { break } } catch {}
  Start-Sleep 1
}
# --protocol http2: works on networks that block Cloudflare's default QUIC/UDP transport
$tun = Start-Process $cf -ArgumentList "tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8000" `
       -RedirectStandardError "artifacts\logs\tunnel.log" -PassThru -NoNewWindow
$url = $null
for ($i = 0; $i -lt 90 -and -not $url; $i++) {
  Start-Sleep 1
  $m = Select-String -Path "artifacts\logs\tunnel.log" -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($m) { $url = $m.Matches[0].Value }
}
Write-Host ""
Write-Host "QualityGuard AI is public at:  $url" -ForegroundColor Green
Write-Host "Local:                         http://127.0.0.1:8000"
Write-Host "Keep this window open. Press Ctrl+C to stop."
try { Wait-Process -Id $app.Id } finally {
  Stop-Process -Id $tun.Id -ErrorAction SilentlyContinue
  Stop-Process -Id $app.Id -ErrorAction SilentlyContinue
}
