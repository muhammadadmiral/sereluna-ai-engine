$ErrorActionPreference = "Stop"

$hf = Join-Path $env:APPDATA "Python\Python313\Scripts\hf.exe"
if (-not (Test-Path $hf)) {
    Write-Host "Installing Hugging Face CLI..."
    python -m pip install -U huggingface_hub
}

$hf = Join-Path $env:APPDATA "Python\Python313\Scripts\hf.exe"
if (-not (Test-Path $hf)) {
    throw "hf.exe not found. Restart terminal, then run this script again."
}

function Invoke-Hf {
    & $hf @args
    if ($LASTEXITCODE -ne 0) {
        throw "hf command failed with exit code $LASTEXITCODE"
    }
}

$username = Read-Host "Hugging Face username"
$spaceName = Read-Host "Space name [sereluna-ai-engine]"
if ([string]::IsNullOrWhiteSpace($spaceName)) {
    $spaceName = "sereluna-ai-engine"
}

$hfTokenFromEnv = $env:HF_TOKEN
if ([string]::IsNullOrWhiteSpace($hfTokenFromEnv)) {
    $hfTokenSecure = Read-Host "HF token with Write permission (starts with hf_)" -AsSecureString
}
$groqKeySecure = Read-Host "GROQ_API_KEY" -AsSecureString
$firebasePath = Read-Host "Path to Firebase service account JSON file"

if (-not (Test-Path $firebasePath)) {
    throw "Firebase service account JSON file not found: $firebasePath"
}

$hfTokenPtr = [IntPtr]::Zero
if (-not [string]::IsNullOrWhiteSpace($hfTokenFromEnv)) {
    $hfToken = $hfTokenFromEnv
} else {
    $hfTokenPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($hfTokenSecure)
}
$groqKeyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($groqKeySecure)

try {
    if ([string]::IsNullOrWhiteSpace($hfToken)) {
        $hfToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($hfTokenPtr)
    }
    $groqKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($groqKeyPtr)
    $repoId = "$username/$spaceName"
    $firebaseJson = Get-Content $firebasePath -Raw | ConvertFrom-Json | ConvertTo-Json -Compress -Depth 100

    Write-Host "Logging in to Hugging Face..."
    Invoke-Hf auth login --token $hfToken --add-to-git-credential

    Write-Host "Creating Docker Space: $repoId"
    Invoke-Hf repos create $repoId `
        --repo-type space `
        --space-sdk docker `
        --private `
        --exist-ok `
        --token $hfToken `
        --env "GROQ_MODEL=llama-3.1-8b-instant" `
        --env "APP_TIMEZONE=Asia/Jakarta" `
        --secrets "GROQ_API_KEY=$groqKey" `
        --secrets "FIREBASE_SERVICE_ACCOUNT_JSON=$firebaseJson"

    Write-Host "Uploading backend files..."
    Invoke-Hf upload $repoId . . `
        --repo-type space `
        --token $hfToken `
        '--exclude=.git/*' `
        '--exclude=.env' `
        '--exclude=.venv/*' `
        '--exclude=venv/*' `
        '--exclude=env/*' `
        '--exclude=__pycache__/*' `
        '--exclude=*/__pycache__/*' `
        '--exclude=*.pyc' `
        '--exclude=*service*.json' `
        '--exclude=*service-account*.json' `
        '--exclude=firebase*.json' `
        --commit-message "Deploy Sereluna FastAPI"

    $spaceHost = "$($username.ToLower())-$($spaceName.ToLower()).hf.space"
    Write-Host ""
    Write-Host "Deploy uploaded. Build logs:"
    Write-Host "https://huggingface.co/spaces/$repoId"
    Write-Host ""
    Write-Host "After build finishes, test:"
    Write-Host "https://$spaceHost/"
    Write-Host "https://$spaceHost/docs"
}
finally {
    if ($hfTokenPtr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($hfTokenPtr)
    }
    if ($groqKeyPtr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($groqKeyPtr)
    }
}
