param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourcePath = Join-Path $ProjectRoot "launcher\NanoBananaPETCTLauncher.cs"

if (-not (Test-Path $SourcePath)) {
    throw "Launcher source not found: $SourcePath"
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $ProjectRoot "NanoBananaPETCT.exe"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$OutputDirectory = Split-Path -Parent $OutputPath
if (-not (Test-Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}

if (Test-Path $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}

$Source = Get-Content -LiteralPath $SourcePath -Raw -Encoding UTF8
Add-Type `
    -TypeDefinition $Source `
    -ReferencedAssemblies @("System.Windows.Forms.dll", "System.Net.Http.dll") `
    -OutputAssembly $OutputPath `
    -OutputType WindowsApplication

Write-Host "Built launcher: $OutputPath"
