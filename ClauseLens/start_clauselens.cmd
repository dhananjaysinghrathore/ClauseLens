@echo off
REM ===== ClauseLens smart start (re-ingest only if PDFs changed) =====

REM 1) Go to your project folder
cd /d "%USERPROFILE%\OneDrive\Documents\ClauseLens"

REM 2) (Optional) activate your virtual env if you use one
REM call ".venv\Scripts\activate"
REM or:  REM call conda activate clauselens

REM 3) Prepare signature file paths
set "SIGFILE=.\index\_data.sig"
set "SIGTMP=%TEMP%\_clauselens_sig.tmp"
if not exist ".\index" mkdir ".\index"

REM 4) Build a signature of all files (path|size|modified) in .\data
powershell -NoLogo -NoProfile -Command ^
  "if (!(Test-Path '.\data')) { New-Item -ItemType Directory -Path '.\data' | Out-Null };" ^
  "Get-ChildItem -Recurse -File '.\data' | ForEach-Object { '{0}|{1}|{2}' -f $_.FullName,$_.Length,$_.LastWriteTimeUtc.Ticks } | Sort-Object | Out-File -Encoding UTF8 '%SIGTMP%';" ^
  "if (Test-Path '%SIGFILE%') { if ((Get-FileHash '%SIGTMP%').Hash -ne (Get-FileHash '%SIGFILE%').Hash) { exit 10 } else { exit 0 } } else { exit 10 }"

REM 5) If signature changed, re-ingest; else skip
if %errorlevel%==10 (
  echo [1/4] Changes detected in data\  -> rebuilding index...
  python ingest.py --data_dir .\data --out_dir .\index
  if errorlevel 1 (
    echo Ingest failed. Press any key to exit.
    pause >nul
    exit /b 1
  )
  copy /Y "%SIGTMP%" "%SIGFILE%" >nul
) else (
  echo [1/4] No changes in data\  -> using existing index.
)

REM 6) Start API and UI in separate windows
echo [2/4] Starting API...
start "ClauseLens API" cmd /k python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

echo [3/4] Starting Streamlit UI...
start "ClauseLens UI" cmd /k python -m streamlit run streamlit_app.py

echo [4/4] Done. Leave both windows open while using ClauseLens.
