param([string]$Destination = (Join-Path $PSScriptRoot 'CARLA_0.9.16'))

$ErrorActionPreference = 'Stop'
$archive = Join-Path $PSScriptRoot 'carla-0.9.16-download.zip'
$url = 'https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_0.9.16.zip'
$expectedSize = 7812252810

if (-not (Test-Path (Join-Path $Destination 'CarlaUE4.exe'))) {
    if (-not (Test-Path $archive) -or (Get-Item $archive).Length -ne $expectedSize) {
        Write-Host 'Downloading CARLA 0.9.16 (about 7.3 GiB); an interrupted download can be resumed.'
        & curl.exe -L --fail --retry 5 --retry-all-errors -C - -o $archive $url
        if ($LASTEXITCODE -ne 0) { throw 'CARLA download failed. Run this script again to resume.' }
    }
    if ((Get-Item $archive).Length -ne $expectedSize) { throw 'CARLA archive has an unexpected size.' }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Write-Host 'Extracting CARLA. This can take several minutes.'
    & tar.exe -xf $archive -C $Destination
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $Destination 'CarlaUE4.exe'))) {
        throw 'CARLA extraction failed or CarlaUE4.exe was not found in the destination.'
    }
}
Write-Host "CARLA installed at $Destination"
