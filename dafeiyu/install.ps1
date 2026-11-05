param([ValidateSet('edge', 'gptsovits')][string]$Voice = 'edge', [switch]$PrepareOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Ask your agent to install prerequisites from AGENTS.md.'
}
$installArgs = @('run', '--no-project', '--python', '3.11', 'python', 'scripts/install.py', '--voice', $Voice)
if ($PrepareOnly) { $installArgs += '--prepare-only' }
& uv @installArgs
if ($LASTEXITCODE -ne 0) { throw "Installation failed ($LASTEXITCODE). Fix the error before launching." }
