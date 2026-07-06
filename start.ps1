# Document Chunking & PostgreSQL Storing Tool — Start Script
# Run this: .\start.ps1
Set-Location "$PSScriptRoot\backend"
Write-Host "`n🗄️ Document Chunking & Storage Tool" -ForegroundColor Cyan
Write-Host "Starting FastAPI server at http://localhost:8000 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow
& "$PSScriptRoot\venv\Scripts\python.exe" run.py

