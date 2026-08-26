$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Data = Join-Path $Root "data"
$Pretrained = Join-Path $Data "pretrained"
$TinyZip = Join-Path $Data "tiny-imagenet-200.zip"
$TinyDirectory = Join-Path $Data "tiny-imagenet-200"
$DeiT = Join-Path $Pretrained "deit_tiny_patch16_224-a1311bcf.pth"

New-Item -ItemType Directory -Force -Path $Data, $Pretrained | Out-Null

if (-not (Test-Path -LiteralPath $TinyDirectory)) {
    & curl.exe -L -C - --retry 20 --retry-all-errors --retry-delay 10 `
        -o $TinyZip "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip"
    if ($LASTEXITCODE -ne 0) { throw "Tiny ImageNet download failed with exit code $LASTEXITCODE" }
    $TinyMd5 = (Get-FileHash -LiteralPath $TinyZip -Algorithm MD5).Hash.ToLowerInvariant()
    if ($TinyMd5 -ne "90528d7ca1a48142e341f4ef8d21d0de") {
        throw "Tiny ImageNet MD5 mismatch: $TinyMd5"
    }
    Expand-Archive -LiteralPath $TinyZip -DestinationPath $Data -Force
}

if (-not (Test-Path -LiteralPath $DeiT)) {
    & curl.exe -L --retry 10 --retry-all-errors --retry-delay 10 `
        -o $DeiT "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    if ($LASTEXITCODE -ne 0) { throw "DeiT download failed with exit code $LASTEXITCODE" }
}
$DeiTSha256 = (Get-FileHash -LiteralPath $DeiT -Algorithm SHA256).Hash.ToLowerInvariant()
if ($DeiTSha256 -ne "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97") {
    throw "DeiT SHA256 mismatch: $DeiTSha256"
}

& (Join-Path $Root ".venv\Scripts\python.exe") -m unittest tests.test_revision_protocol -v
if ($LASTEXITCODE -ne 0) { throw "External asset validation failed" }
