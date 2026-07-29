#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

$cloudFolder = -join [char[]]@(
    0x8FC5,
    0x96F7,
    0x4E91,
    0x76D8
)
$sourceRoot = Join-Path `
    ("E:\" + $cloudFolder) `
    "ffmpeg-8.1.2-essentials_build\ffmpeg-8.1.2-essentials_build"
$installRoot = "C:\Program Files\FFmpeg"
$binPath = Join-Path $installRoot "bin"
$sourceFfmpeg = Join-Path $sourceRoot "bin\ffmpeg.exe"
$sourceFfprobe = Join-Path $sourceRoot "bin\ffprobe.exe"

try {
    Write-Host "1/4 Checking the downloaded files..." -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $sourceFfmpeg)) {
        throw "ffmpeg.exe was not found in: $sourceRoot"
    }
    if (-not (Test-Path -LiteralPath $sourceFfprobe)) {
        throw "ffprobe.exe was not found in: $sourceRoot"
    }
    if (Test-Path -LiteralPath $installRoot) {
        throw "The install directory already exists: $installRoot"
    }

    Write-Host "2/4 Copying files to Program Files..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $installRoot | Out-Null
    Get-ChildItem -LiteralPath $sourceRoot -Force |
        Copy-Item -Destination $installRoot -Recurse -Force

    $installedFfmpeg = Join-Path $binPath "ffmpeg.exe"
    $installedFfprobe = Join-Path $binPath "ffprobe.exe"
    if (
        -not (Test-Path -LiteralPath $installedFfmpeg) -or
        -not (Test-Path -LiteralPath $installedFfprobe)
    ) {
        throw "The FFmpeg executables were not found after copying."
    }

    Write-Host "3/4 Adding FFmpeg to the machine PATH..." -ForegroundColor Cyan
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

    Write-Host "4/4 Verifying FFmpeg and FFprobe..." -ForegroundColor Cyan
    $ffmpegVersion = & $installedFfmpeg -version 2>&1 |
        Select-Object -First 1
    $ffprobeVersion = & $installedFfprobe -version 2>&1 |
        Select-Object -First 1

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
