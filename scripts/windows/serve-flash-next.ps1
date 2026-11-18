<#
.SYNOPSIS
  Serve Qwen3.8-Flash-Next (EXL3 4.05 bpw) to the LAN: OpenAI-compatible API, bearer token, 128K context.

.DESCRIPTION
  Wraps `scripts/windows/kyojin_win.py serve` (-> tools/qwen/serve.py). Same shape as serve-lan.ps1 (the 122B runner).
  - Listens on 0.0.0.0: other machines on the LAN reach http://<this-PC-ip>:<port>/v1
  - Every route except GET /health needs `Authorization: Bearer <token>` (KYOJIN_API_KEY, opt-in middleware in serve.py).
    Token: -Token <text> if given, else the first line of -TokenFile, else a new simple token `qwen-flash-next-<8 hex>` is
    generated and saved to -TokenFile. It is short on purpose (easy to type); on a LAN you do not trust, pass a longer -Token.
  - Context 131072 tokens (fp16 KV: 12 full-attention layers x 2 KV heads x 256 dim = 24 KiB/token, ~3 GiB at 128K, on top
    of ~64 GiB of weights; the 36 GiB n-gram table streams from the SSD).
  - Chat template: serve.py's built-in one (the model's own plus a `medium` effort instruction), so reasoning_effort works.
  - Thinking: off unless the request turns it on (-Thinking makes it the default). -Effort sets the default effort.
  - SSE keep-alive every 15 s so a client does not drop the connection during a long prefill.
  - One session (one request at a time); the prompt cache keeps the shared prefix.

.EXAMPLE
  .\scripts\windows\serve-flash-next.ps1
.EXAMPLE
  .\scripts\windows\serve-flash-next.ps1 -Token qwen-flash-next-2026 -OpenFirewall   # firewall: once, elevated PowerShell
.EXAMPLE
  .\scripts\windows\serve-flash-next.ps1 -Thinking -Effort low -Port 8080
#>
[CmdletBinding()]
param(
    [string]$Model       = 'C:\models\flash-next-4p05',
    [int]   $Port        = 8000,
    [int]   $Ctx         = 131072,
    [string]$Token       = '',
    [string]$TokenFile   = (Join-Path $env:USERPROFILE '.kyojin\flash-next-token.txt'),
    [string]$ModelId     = 'qwen3.8-flash-next',
    [string]$SlotDir     = (Join-Path $env:USERPROFILE '.kyojin\slots-flash-next'),
    [ValidateSet('off', 'mix', 'mtp')]
    [string]$DraftPolicy = 'mix',
    [ValidateSet('low', 'medium', 'xhigh')]
    [string]$Effort      = 'medium',
    [int]   $MaxReply    = 16384,
    [switch]$Thinking,
    [switch]$OpenFirewall
)

$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$py   = Join-Path $repo '.venv-win\Scripts\python.exe'
if (-not (Test-Path $py))    { throw "No venv at $py. Run: uv run --no-project python scripts/windows/kyojin_win.py setup" }
if (-not (Test-Path $Model)) { throw "Model directory not found: $Model (hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6 --local-dir $Model)" }

# --- bearer token -------------------------------------------------------------------------------------------------
if (-not $Token) {
    if (-not (Test-Path $TokenFile)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $TokenFile -Parent) | Out-Null
        $bytes = New-Object byte[] 4
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $new = 'qwen-flash-next-' + ([BitConverter]::ToString($bytes) -replace '-', '').ToLower()
        Set-Content -Path $TokenFile -Value $new -NoNewline -Encoding ascii
        icacls $TokenFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null   # readable by this user only
        Write-Host "Generated a new API token and saved it to $TokenFile" -ForegroundColor Yellow
    }
    $Token = (Get-Content -Raw $TokenFile).Trim()
}
if ($Token.Length -lt 8) { throw "The token is shorter than 8 characters." }
if ($Token -match '\s')  { throw "The token must not contain spaces." }
$env:KYOJIN_API_KEY = $Token

# --- optional firewall rule (needs an elevated shell; Private profile, local subnet only) ------------------------------
if ($OpenFirewall) {
    $name = "Kyojin Flash-Next server (TCP $Port)"
    Remove-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port `
        -Action Allow -Profile Private -RemoteAddress LocalSubnet | Out-Null
    Write-Host "Firewall: inbound TCP $Port allowed on the Private profile from the local subnet." -ForegroundColor Green
}

# --- environment ----------------------------------------------------------------------------------------------------
$env:KYOJIN_SSE_PING_S = '15'
$env:PYTHONUTF8        = '1'

# --- show how to connect ------------------------------------------------------------------------------------------
$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
    Select-Object -ExpandProperty IPAddress
Write-Host ''
Write-Host "Kyojin Flash-Next server: $ModelId, context $Ctx, drafting $DraftPolicy, thinking default: $(if ($Thinking) {"on, effort $Effort"} else {'off'})"
foreach ($ip in $ips) { Write-Host "  Base URL : http://${ip}:$Port/v1" }
Write-Host "  API key  : $(if ($PSBoundParameters.ContainsKey('Token')) {'the -Token you passed'} else {$TokenFile})  (send as 'Authorization: Bearer <token>')"
Write-Host "  Model id : $ModelId"
Write-Host "  Test     : curl.exe http://<ip>:$Port/v1/models -H `"Authorization: Bearer <token>`""
Write-Host 'The port opens only when the model is loaded (1-3 min; GET /health shows progress without a token).'
Write-Host ''

# --- serve.py flags (after `--`; they come last, so they override the driver's defaults) ---------------------------------
New-Item -ItemType Directory -Force -Path $SlotDir | Out-Null   # POST /slots/0?action=save does not create the directory
$extra = @(
    '--host', '0.0.0.0',
    '--draft-policy', $DraftPolicy,
    '--slot-save-path', $SlotDir,
    '--model-id', $ModelId,
    '--default-max-tokens', "$MaxReply",
    '--default-reasoning-effort', $Effort
)
if (-not $Thinking) { $extra += '--no-thinking' }

Push-Location $repo
try {
    & $py -X utf8 scripts\windows\kyojin_win.py serve --model $Model --port $Port --ctx $Ctx -- @extra
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
