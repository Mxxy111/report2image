param(
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PETCT_WEB_HOST = $HostName
$env:PETCT_WEB_PORT = [string]$Port

python -m webapp.main
