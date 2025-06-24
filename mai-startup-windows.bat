@echo off
echo 🚀 Starting MAI FastAPI server...
start cmd /k "uvicorn app:app --host 0.0.0.0 --port 5000 --reload"

timeout /t 3 >nul
echo 🌐 Starting Cloudflare Tunnel...
start cmd /k "cloudflared tunnel --url http://localhost:5000"

echo ✅ Mai is now available via your Cloudflare tunnel.
pause