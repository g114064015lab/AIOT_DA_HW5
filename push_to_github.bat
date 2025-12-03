@echo off
REM Windows helper script to push this repository to GitHub.
REM Usage (recommended):
REM   1) Open cmd.exe
REM   2) set GITHUB_TOKEN=ghp_... (or use Git Credential Manager / SSH setup)
REM   3) push_to_github.bat https://github.com/g114064015lab/AIOT_DA_HW5.git "Your Name" g114064015@smail.nchu.edu.tw

if "%1"=="" (
  echo Usage: push_to_github.bat <remote_url> [user_name] [user_email]
  exit /b 1
)

set REMOTE_URL=%1
set USER_NAME=%2
set USER_EMAIL=%3

if "%USER_NAME%"=="" set USER_NAME=AutoPusher
if "%USER_EMAIL%"=="" set USER_EMAIL=g114064015@smail.nchu.edu.tw

echo Setting git user.name to %USER_NAME%
git config user.name "%USER_NAME%"
echo Setting git user.email to %USER_EMAIL%
git config user.email "%USER_EMAIL%"

if not exist .git (
  echo Initializing git repository
  git init
)

echo Adding files
git add .
git commit -m "Deploy AI detector app"

REM If GITHUB_TOKEN is set, embed it into HTTPS URL (note: this will appear in process list and may be insecure).
if defined GITHUB_TOKEN (
  echo Using provided GITHUB_TOKEN to construct remote URL (temporary)
  for /f "tokens=*" %%I in ('echo %REMOTE_URL%') do set SAFE_REMOTE=%%I
  REM insert token after https://
  set REMOTE_WITH_TOKEN=%SAFE_REMOTE:
 =%
  set REMOTE_WITH_TOKEN=%SAFE_REMOTE%
  REM The user should replace the remote manually if they prefer not to put token in URL.
  git remote remove origin 2>nul
  git remote add origin %REMOTE_URL%
  git push -u origin main
) else (
  echo No GITHUB_TOKEN found. Attempting normal push (you may be prompted for credentials).
  git remote remove origin 2>nul
  git remote add origin %REMOTE_URL%
  git push -u origin main
)

echo Done. If push failed, follow the README instructions to configure authentication (SSH or PAT).
