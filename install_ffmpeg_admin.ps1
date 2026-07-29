#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$downloadUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$checksumUrl = "$downloadUrl.sha256"
$installRoot = "C:\Program Files\FFmpeg"
$binPath = Join-Path $installRoot "bin"
$temporaryRoot = Join-Path `
    ([System.IO.Path]::GetTempPath()) `
    ("ffmpeg-install-" + [Guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $temporaryRoot "ffmpeg-release-essentials.zip"
$checksumPath = "$archivePath.sha256"
$extractRoot = Join-Path $temporaryRoot "extracted"

try {
    if (Test-Path -LiteralPath $installRoot) {
        throw "The install directory already exists: $installRoot"
    }

    Write-Host "1/5 Downloading FFmpeg..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
    Invoke-WebRequest -Uri $checksumUrl -OutFile $checksumPath

    Write-Host "2/5 Verifying SHA-256..." -ForegroundColor Cyan
    $expectedHash = (
        (Get-Content -LiteralPath $checksumPath -Raw).Trim() -split "\s+"
    )[0].ToLowerInvariant()
    $actualHash = (
        Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Checksum mismatch. Expected $expectedHash, got $actualHash"
    }

    Write-Host "3/5 Extracting and installing..." -ForegroundColor Cyan
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractRoot
    $sourceRoot = Get-ChildItem -LiteralPath $extractRoot -Directory |
        Select-Object -First 1
    if ($null -eq $sourceRoot) {
        throw "The downloaded archive has an unexpected structure."
    }
    New-Item -ItemType Directory -Path $installRoot | Out-Null
    Copy-Item `
        -Path (Join-Path $sourceRoot.FullName "*") `
        -Destination $installRoot `
        -Recurse

    $ffmpegExe = Join-Path $binPath "ffmpeg.exe"
    $ffprobeExe = Join-Path $binPath "ffprobe.exe"
    if (
        -not (Test-Path -LiteralPath $ffmpegExe) -or
        -not (Test-Path -LiteralPath $ffprobeExe)
    ) {
        throw "ffmpeg.exe or ffprobe.exe was not found after extraction."
    }

    Write-Host "4/5 Updating the machine PATH..." -ForegroundColor Cyan
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $pathEntries = @(
        $machinePath -split ";" |
            ForEach-Object { $_.Trim().TrimEnd("\") } |
            Where-Object { $_ }
    )
    $alreadyPresent = $pathEntries |
        Where-Object {
            [string]::Equals(
                $_,
                $binPath,
                [StringComparison]::OrdinalIgnoreCase
            )
        }
    if (-not $alreadyPresent) {
        $newMachinePath = $machinePath.TrimEnd(";") + ";" + $binPath
        [Environment]::SetEnvironmentVariable(
            "Path",
            $newMachinePath,
            "Machine"
        )
    }

    Write-Host "5/5 Verifying the installation..." -ForegroundColor Cyan
    $ffmpegVersion = & $ffmpegExe -version 2>&1 | Select-Object -First 1
    $ffprobeVersion = & $ffprobeExe -version 2>&1 | Select-Object -First 1

    Write-Host ""
    Write-Host "FFmpeg was installed successfully." -ForegroundColor Green
    Write-Host $ffmpegVersion
    Write-Host $ffprobeVersion
    Write-Host ""
    Write-Host "Install directory: $installRoot"
    Write-Host "Machine PATH entry: $binPath"
    Write-Host "Reopen terminals and the detector GUI to use the new PATH."
}
catch {
    Write-Host ""
    Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    if (
        $temporaryRoot.StartsWith(
            [System.IO.Path]::GetTempPath(),
            [StringComparison]::OrdinalIgnoreCase
        ) -and
        (Test-Path -LiteralPath $temporaryRoot)
    ) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
