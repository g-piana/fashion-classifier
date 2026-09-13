# deploy.ps1
# ==========
# Prepares model artifacts, uploads to S3, builds the container image,
# pushes to ECR, and deploys (or updates) the Lambda function.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker Desktop running
#   - python lambda-inference/test_local.py passes
#
# Usage:
#   cd E:\fashion-classifier
#   .\lambda-inference\deploy.ps1 `
#       -BucketName  "my-fashion-models" `
#       -Region      "eu-west-1" `
#       -CkptPath    "E:\fashion-data\weights\jackets\01\best.ckpt" `
#       -NormPath    "E:\fashion-data\weights\jackets\01\normalization.npy" `
#       -Classes     "biker,blazer,bomber,fur jacket,parka" `
#       -Domain      "jackets" `
#       -Run         "01"

param(
    [Parameter(Mandatory)][string]$BucketName,
    [Parameter(Mandatory)][string]$Region,
    [Parameter(Mandatory)][string]$CkptPath,
    [Parameter(Mandatory)][string]$NormPath,
    [Parameter(Mandatory)][string]$Classes,
    [Parameter(Mandatory)][string]$Domain,
    [Parameter(Mandatory)][string]$Run,
    [string]$MemoryMB   = "1024",
    [string]$TimeoutSec = "60"
)

$ErrorActionPreference = "Stop"

$AccountId   = (aws sts get-caller-identity --query Account --output text)
$EcrRegistry = "${AccountId}.dkr.ecr.${Region}.amazonaws.com"
$ImageName   = "fashion-classifier-inference"
$ImageTag    = "${Domain}-${Run}"
$FullImage   = "${EcrRegistry}/${ImageName}:${ImageTag}"
$LambdaName  = "fashion-inference-${Domain}"
$S3KeyCkpt   = "${Domain}/${Run}/best_weights.pt"
$S3KeyNorm   = "${Domain}/${Run}/normalization.npy"

Write-Host "`n=== Step 1: Upload model artifacts to S3 ===" -ForegroundColor Cyan

# Ensure bucket exists
aws s3api head-bucket --bucket $BucketName 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating bucket $BucketName..."
    aws s3api create-bucket `
        --bucket $BucketName `
        --region $Region `
        --create-bucket-configuration LocationConstraint=$Region
}

aws s3 cp $CkptPath "s3://${BucketName}/${S3KeyCkpt}"
aws s3 cp $NormPath "s3://${BucketName}/${S3KeyNorm}"
Write-Host "✓ Artifacts uploaded"

Write-Host "`n=== Step 2: Build container image ===" -ForegroundColor Cyan

Set-Location lambda-inference
docker build -t "${ImageName}:${ImageTag}" .
Write-Host "✓ Image built"

Write-Host "`n=== Step 3: Push to ECR ===" -ForegroundColor Cyan

# Create ECR repo if needed
aws ecr describe-repositories --repository-names $ImageName --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --repository-name $ImageName --region $Region
}

# Authenticate Docker to ECR
aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin $EcrRegistry

docker tag "${ImageName}:${ImageTag}" $FullImage
docker push $FullImage
Write-Host "✓ Image pushed: $FullImage"

Write-Host "`n=== Step 4: Deploy Lambda ===" -ForegroundColor Cyan

$EnvVars = "Variables={" +
    "MODEL_BUCKET=${BucketName}," +
    "MODEL_KEY_CKPT=${S3KeyCkpt}," +
    "MODEL_KEY_NORM=${S3KeyNorm}," +
    "MODEL_CLASSES=${Classes}," +
    "MODEL_IMAGE_SIZE=224" +
    "}"

# Check if function exists
aws lambda get-function --function-name $LambdaName --region $Region 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "Updating existing Lambda function..."
    aws lambda update-function-code `
        --function-name $LambdaName `
        --image-uri $FullImage `
        --region $Region
    aws lambda update-function-configuration `
        --function-name $LambdaName `
        --memory-size $MemoryMB `
        --timeout $TimeoutSec `
        --environment $EnvVars `
        --region $Region
} else {
    Write-Host "Creating new Lambda function..."
    $RoleArn = "arn:aws:iam::${AccountId}:role/lambda-fashion-inference-role"
    aws lambda create-function `
        --function-name $LambdaName `
        --package-type Image `
        --code "ImageUri=${FullImage}" `
        --role $RoleArn `
        --memory-size $MemoryMB `
        --timeout $TimeoutSec `
        --environment $EnvVars `
        --region $Region
    Write-Host "NOTE: IAM role '${RoleArn}' must exist with AWSLambdaBasicExecutionRole + S3 read"
}

Write-Host "✓ Lambda deployed: $LambdaName"

Write-Host "`n=== Step 5: Smoke test ===" -ForegroundColor Cyan
Set-Location ..
python lambda-inference/test_invoke.py `
    --function $LambdaName `
    --region $Region `
    --image "E:\fashion-data\01-RAW\jackets_img\test_image.jpg"

Write-Host "`n✓ Done. Lambda function: $LambdaName" -ForegroundColor Green
