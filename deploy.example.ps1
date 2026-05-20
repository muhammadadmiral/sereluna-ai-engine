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

# ==========================================
# KONFIGURASI DEPLOYMENT (ISI MANUAL DI SINI)
# ==========================================
$username = "YOUR_HF_USERNAME_HERE"
$spaceName = "YOUR_SPACE_NAME_HERE"
$hfToken = "YOUR_HF_TOKEN_HERE"
$groqKey = "YOUR_GROQ_API_KEY_HERE"
$firebasePath = "google-service.json" # Atur path jika berbeda
# ==========================================

if (-not (Test-Path $firebasePath)) {
    throw "Firebase service account JSON file not found: $firebasePath"
}

$repoId = "$username/$spaceName"
$firebaseAccount = Get-Content $firebasePath -Raw | ConvertFrom-Json
$firebaseJson = $firebaseAccount | ConvertTo-Json -Compress -Depth 100
$firebaseProjectId = $firebaseAccount.project_id
if ([string]::IsNullOrWhiteSpace($firebaseProjectId)) {
    throw "Firebase service account JSON does not contain project_id."
}

$tempSecretsFile = Join-Path $env:TEMP "sereluna-hf-secrets-$([guid]::NewGuid().ToString('N')).env"
$tempSecretsContent = @(
    "GROQ_API_KEY=$groqKey"
    "FIREBASE_SERVICE_ACCOUNT_JSON=$firebaseJson"
) -join "`n"

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
    --env "APP_TIMEZONE=Asia/Jakarta"

Write-Host "Setting Space secrets..."
Set-Content -LiteralPath $tempSecretsFile -Value $tempSecretsContent -Encoding utf8
try {
    Invoke-Hf spaces secrets add $repoId `
        --secrets-file $tempSecretsFile `
        --token $hfToken
}
finally {
    if (Test-Path $tempSecretsFile) {
        Remove-Item -LiteralPath $tempSecretsFile -Force
    }
}

Write-Host "Setting Firebase project variables: $firebaseProjectId"
Invoke-Hf spaces variables add $repoId `
    -e "FIREBASE_PROJECT_ID=$firebaseProjectId" `
    -e "GOOGLE_CLOUD_PROJECT=$firebaseProjectId" `
    -e "GCLOUD_PROJECT=$firebaseProjectId" `
    --token $hfToken

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