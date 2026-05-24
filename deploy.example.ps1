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
$nvidiaKey = "YOUR_NVIDIA_NGC_API_KEY_HERE"
$groqKey = "YOUR_GROQ_API_KEY_HERE"
$guardianKey = "YOUR_GUARDIAN_API_KEY_HERE"
$firebasePath = "google-service.json" # Atur path jika berbeda
$firebaseStorageBucket = "YOUR_FIREBASE_STORAGE_BUCKET_HERE"
$doctorGuardrailInstruction = ""
$doctorDirectReply = ""
# ==========================================

if (-not (Test-Path $firebasePath)) {
    throw "Firebase service account JSON file not found: $firebasePath"
}

if ([string]::IsNullOrWhiteSpace($nvidiaKey) -or -not $nvidiaKey.StartsWith("nvapi-")) {
    throw "NVIDIA API key is missing or invalid. Fill `$nvidiaKey with a fresh nvapi-... key."
}

if ([string]::IsNullOrWhiteSpace($firebaseStorageBucket) -or $firebaseStorageBucket -like "YOUR_*") {
    throw "Firebase Storage bucket is missing. Fill `$firebaseStorageBucket, for example sereluna2024.appspot.com."
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
    "NVIDIA_API_KEY=$nvidiaKey"
    "GROQ_API_KEY=$groqKey"
    "GUARDIAN_API_KEY=$guardianKey"
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
    --env "LLM_PROVIDER_MODE=fallback" `
    --env "NVIDIA_MODEL=deepseek-ai/deepseek-v4-flash" `
    --env "NVIDIA_FAST_MODEL=meta/llama-3.1-8b-instruct" `
    --env "NVIDIA_VISION_MODEL=google/gemma-3n-e2b-it" `
    --env "NVIDIA_THINKING=false" `
    --env "NVIDIA_TOP_P=0.95" `
    --env "GROQ_MODEL=llama-3.3-70b-versatile" `
    --env "GROQ_BACKUP_MODEL=moonshotai/kimi-k2-instruct" `
    --env "GROQ_FAST_MODEL=llama-3.1-8b-instant" `
    --env "APP_TIMEZONE=Asia/Jakarta" `
    --env "DOCTOR_MENU_GUARDRAIL_INSTRUCTION=$doctorGuardrailInstruction" `
    --env "DOCTOR_DIRECT_REPLY=$doctorDirectReply"

Write-Host "Setting Space secrets..."
Set-Content -LiteralPath $tempSecretsFile -Value $tempSecretsContent -Encoding ascii
try {
    Invoke-Hf spaces secrets add $repoId `
        --secrets-file $tempSecretsFile `
        --token $hfToken
    Invoke-Hf spaces secrets add $repoId `
        -s "NVIDIA_API_KEY=$nvidiaKey" `
        --token $hfToken
}
finally {
    if (Test-Path $tempSecretsFile) {
        Remove-Item -LiteralPath $tempSecretsFile -Force
    }
}

Write-Host "Setting Firebase project variables: $firebaseProjectId"
Invoke-Hf spaces variables add $repoId `
    -e "LLM_PROVIDER_MODE=fallback" `
    -e "NVIDIA_MODEL=deepseek-ai/deepseek-v4-flash" `
    -e "NVIDIA_FAST_MODEL=meta/llama-3.1-8b-instruct" `
    -e "NVIDIA_VISION_MODEL=google/gemma-3n-e2b-it" `
    -e "NVIDIA_THINKING=false" `
    -e "NVIDIA_TOP_P=0.95" `
    -e "NVIDIA_MAX_TOKENS=320" `
    -e "NVIDIA_TIMEOUT_SECONDS=60" `
    -e "GROQ_MODEL=llama-3.3-70b-versatile" `
    -e "GROQ_BACKUP_MODEL=moonshotai/kimi-k2-instruct" `
    -e "GROQ_FAST_MODEL=llama-3.1-8b-instant" `
    -e "FIREBASE_PROJECT_ID=$firebaseProjectId" `
    -e "FIREBASE_STORAGE_BUCKET=$firebaseStorageBucket" `
    -e "GOOGLE_CLOUD_PROJECT=$firebaseProjectId" `
    -e "GCLOUD_PROJECT=$firebaseProjectId" `
    -e "DOCTOR_MENU_GUARDRAIL_INSTRUCTION=$doctorGuardrailInstruction" `
    -e "DOCTOR_DIRECT_REPLY=$doctorDirectReply" `
    --token $hfToken

Write-Host "Uploading backend files..."
Invoke-Hf upload $repoId . . `
    --repo-type space `
    --token $hfToken `
    '--exclude=.git/*' `
    '--exclude=.env' `
    '--exclude=*.pdf' `
    '--exclude=*.docx' `
    '--exclude=*.doc' `
    '--exclude=*.pptx' `
    '--exclude=*.ppt' `
    '--exclude=data/doctors.local.json' `
    '--exclude=data/doctors.*.local.json' `
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
