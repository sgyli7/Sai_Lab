[CmdletBinding()]
param(
    [int]$TargetProcessId = 0,
    [string]$PlayerPath = "",
    [string]$OutputDirectory = "",
    [ValidateRange(4, 30)]
    [int]$FramesPerSecond = 8,
    [switch]$KeepPlayerOpen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($PlayerPath)) {
    $PlayerPath = Join-Path $ProjectRoot "Builds\Windows64\AgenticRobotGame.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "artifacts\visual-acceptance\interactive"
}
$PlayerPath = [System.IO.Path]::GetFullPath($PlayerPath)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class MicroDuckVisualAcceptanceNative
{
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr window, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr window, IntPtr destination, uint flags);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr window);

    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr window, uint message, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern void keybd_event(byte virtualKey, byte scanCode, uint flags, UIntPtr extraInfo);

    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint flags, int dx, int dy, int data, UIntPtr extraInfo);
}
'@

function Send-KeyDown {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Window,
        [Parameter(Mandatory = $true)][byte]$VirtualKey
    )
    [MicroDuckVisualAcceptanceNative]::SetForegroundWindow($Window) | Out-Null
    [MicroDuckVisualAcceptanceNative]::keybd_event($VirtualKey, 0, 0, [UIntPtr]::Zero)
    [MicroDuckVisualAcceptanceNative]::PostMessage(
        $Window, 0x0100, [IntPtr]([int]$VirtualKey), [IntPtr]::Zero) | Out-Null
}

function Send-KeyUp {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Window,
        [Parameter(Mandatory = $true)][byte]$VirtualKey
    )
    [MicroDuckVisualAcceptanceNative]::keybd_event($VirtualKey, 0, 2, [UIntPtr]::Zero)
    [MicroDuckVisualAcceptanceNative]::PostMessage(
        $Window, 0x0101, [IntPtr]([int]$VirtualKey), [IntPtr]::Zero) | Out-Null
}

function Send-KeyPress {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Window,
        [Parameter(Mandatory = $true)][byte]$VirtualKey
    )
    Send-KeyDown -Window $Window -VirtualKey $VirtualKey
    Start-Sleep -Milliseconds 80
    Send-KeyUp -Window $Window -VirtualKey $VirtualKey
}

function Send-MouseOrbit {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Window,
        [Parameter(Mandatory = $true)][int]$DeltaX,
        [Parameter(Mandatory = $true)][int]$DeltaY
    )
    $rect = New-Object MicroDuckVisualAcceptanceNative+RECT
    if (-not [MicroDuckVisualAcceptanceNative]::GetWindowRect($Window, [ref]$rect)) {
        throw "Could not position the orbit gesture in the Player window."
    }
    $centerX = [int](($rect.Left + $rect.Right) / 2)
    $centerY = [int](($rect.Top + $rect.Bottom) / 2)
    [MicroDuckVisualAcceptanceNative]::SetForegroundWindow($Window) | Out-Null
    [MicroDuckVisualAcceptanceNative]::SetCursorPos($centerX, $centerY) | Out-Null
    [MicroDuckVisualAcceptanceNative]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    [MicroDuckVisualAcceptanceNative]::mouse_event(0x0001, $DeltaX, $DeltaY, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 80
    [MicroDuckVisualAcceptanceNative]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
}

function Send-MouseWheel {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Window,
        [Parameter(Mandatory = $true)][int]$Delta
    )
    [MicroDuckVisualAcceptanceNative]::SetForegroundWindow($Window) | Out-Null
    [MicroDuckVisualAcceptanceNative]::mouse_event(0x0800, 0, 0, $Delta, [UIntPtr]::Zero)
}

function Save-WindowFrame {
    param(
        [Parameter(Mandatory = $true)][IntPtr]$Window,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $rect = New-Object MicroDuckVisualAcceptanceNative+RECT
    if (-not [MicroDuckVisualAcceptanceNative]::GetWindowRect($Window, [ref]$rect)) {
        throw "Could not read the Player window bounds."
    }
    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    if ($width -le 0 -or $height -le 0) {
        throw "Player window has invalid bounds: ${width}x${height}."
    }

    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $deviceContext = $graphics.GetHdc()
        try {
            if (-not [MicroDuckVisualAcceptanceNative]::PrintWindow($Window, $deviceContext, 2)) {
                throw "PrintWindow failed for the Player."
            }
        } finally {
            $graphics.ReleaseHdc($deviceContext)
        }
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$session = [DateTimeOffset]::Now.ToString("yyyyMMdd-HHmmss")
$sessionDirectory = Join-Path $OutputDirectory $session
$frameDirectory = Join-Path $sessionDirectory "frames"
New-Item -ItemType Directory -Force -Path $frameDirectory | Out-Null
$logPath = Join-Path $sessionDirectory "player.log"

$launchedHere = $false
if ($TargetProcessId -gt 0) {
    $player = Get-Process -Id $TargetProcessId -ErrorAction Stop
} else {
    if (-not (Test-Path -LiteralPath $PlayerPath -PathType Leaf)) {
        throw "Windows Player not found: $PlayerPath"
    }
    $player = Start-Process -FilePath $PlayerPath `
        -ArgumentList @(
            "-screen-width", "1280",
            "-screen-height", "720",
            "-screen-fullscreen", "0",
            "-logFile", $logPath
        ) `
        -WorkingDirectory (Split-Path -Parent $PlayerPath) `
        -PassThru
    $launchedHere = $true
}

$deadline = [DateTimeOffset]::Now.AddSeconds(20)
do {
    Start-Sleep -Milliseconds 100
    $player.Refresh()
} while ($player.MainWindowHandle -eq [IntPtr]::Zero -and [DateTimeOffset]::Now -lt $deadline)
if ($player.MainWindowHandle -eq [IntPtr]::Zero) {
    throw "Windows Player did not create a visible window."
}

$window = $player.MainWindowHandle
[MicroDuckVisualAcceptanceNative]::SetForegroundWindow($window) | Out-Null
if ($launchedHere) {
    # MainWindowHandle becomes available during the Tuanjie splash screen. Give
    # the playable scene time to load before starting the evidence clock so the
    # first policy/reset commands cannot be swallowed by startup.
    Start-Sleep -Seconds 4
    $player.Refresh()
    if ($player.HasExited) {
        throw "Windows Player exited before visual acceptance could begin."
    }
    $window = $player.MainWindowHandle
    [MicroDuckVisualAcceptanceNative]::SetForegroundWindow($window) | Out-Null
}

$events = @(
    [pscustomobject]@{ At = 0.2; Name = "select standing policy 2 on flat_plaza"; Kind = "press"; Key = [byte]0x32 },
    [pscustomobject]@{ At = 0.5; Name = "reset upright on flat_plaza"; Kind = "press"; Key = [byte]0x52 },
    [pscustomobject]@{ At = 1.0; Name = "camera Side to Rear"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 2.0; Name = "camera Rear to Top"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 3.0; Name = "camera Top to Showcase"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 4.0; Name = "camera Showcase to Side"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 4.3; Name = "orbit follow camera with LMB"; Kind = "orbit"; DeltaX = 90; DeltaY = -25 },
    [pscustomobject]@{ At = 4.7; Name = "zoom follow camera with mouse wheel"; Kind = "wheel"; Delta = 120 },
    [pscustomobject]@{ At = 5.0; Name = "enter camera FreeFly"; Kind = "press"; Key = [byte]0x09 },
    [pscustomobject]@{ At = 5.3; Name = "move FreeFly camera forward"; Kind = "down"; Key = [byte]0x57 },
    [pscustomobject]@{ At = 6.2; Name = "stop FreeFly camera"; Kind = "up"; Key = [byte]0x57 },
    [pscustomobject]@{ At = 6.4; Name = "leave camera FreeFly"; Kind = "press"; Key = [byte]0x09 },
    [pscustomobject]@{ At = 6.7; Name = "focus camera on Duck"; Kind = "press"; Key = [byte]0x46 },
    [pscustomobject]@{ At = 7.5; Name = "terrain upstream_pyramid_stairs"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 9.8; Name = "terrain upstream_random_grid"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 10.1; Name = "select walking policy 1"; Kind = "press"; Key = [byte]0x31 },
    [pscustomobject]@{ At = 10.5; Name = "walk across upstream_random_grid"; Kind = "down"; Key = [byte]0x57 },
    [pscustomobject]@{ At = 11.7; Name = "stop on upstream_random_grid"; Kind = "up"; Key = [byte]0x57 },
    [pscustomobject]@{ At = 12.4; Name = "terrain upstream_pyramid_slope"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 14.8; Name = "terrain upstream_roller_slope"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 15.1; Name = "select roller policy 7"; Kind = "press"; Key = [byte]0x37 },
    [pscustomobject]@{ At = 17.5; Name = "terrain rock_steps"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 17.8; Name = "return to legged walking policy 1"; Kind = "press"; Key = [byte]0x31 },
    [pscustomobject]@{ At = 20.2; Name = "terrain stairs_bridge"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 21.45; Name = "camera handoff Side to Rear"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 21.65; Name = "camera handoff Rear to Top"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 21.85; Name = "camera handoff Top to Showcase"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 22.05; Name = "restore camera Side preset for handoff"; Kind = "press"; Key = [byte]0x43 },
    [pscustomobject]@{ At = 22.3; Name = "return to flat_plaza for handoff"; Kind = "press"; Key = [byte]0x54 },
    [pscustomobject]@{ At = 22.55; Name = "select standing policy 2 for handoff"; Kind = "press"; Key = [byte]0x32 },
    [pscustomobject]@{ At = 22.8; Name = "final upright reset on flat_plaza"; Kind = "press"; Key = [byte]0x52 }
)
$durationSeconds = 24.0
$frameInterval = 1.0 / $FramesPerSecond
$frameCount = [int][Math]::Ceiling($durationSeconds * $FramesPerSecond)
$nextEvent = 0
$eventEvidence = [System.Collections.Generic.List[object]]::new()
$heldKeys = [System.Collections.Generic.HashSet[byte]]::new()
$clock = [System.Diagnostics.Stopwatch]::StartNew()

try {
    for ($frameIndex = 0; $frameIndex -lt $frameCount; $frameIndex++) {
        $targetTime = $frameIndex * $frameInterval
        while ($clock.Elapsed.TotalSeconds -lt $targetTime) {
            Start-Sleep -Milliseconds 5
        }

        while ($nextEvent -lt $events.Count -and $events[$nextEvent].At -le $clock.Elapsed.TotalSeconds) {
            $event = $events[$nextEvent]
            [MicroDuckVisualAcceptanceNative]::SetForegroundWindow($window) | Out-Null
            switch ($event.Kind) {
                "press" { Send-KeyPress -Window $window -VirtualKey $event.Key }
                "down" {
                    $heldKeys.Add($event.Key) | Out-Null
                    Send-KeyDown -Window $window -VirtualKey $event.Key
                }
                "up" {
                    $heldKeys.Remove($event.Key) | Out-Null
                    Send-KeyUp -Window $window -VirtualKey $event.Key
                }
                "orbit" {
                    Send-MouseOrbit -Window $window -DeltaX $event.DeltaX -DeltaY $event.DeltaY
                }
                "wheel" {
                    Send-MouseWheel -Window $window -Delta $event.Delta
                }
            }
            $eventEvidence.Add([pscustomobject]@{
                name = $event.Name
                kind = $event.Kind
                virtualKey = if ($null -ne $event.Key) { [int]$event.Key } else { $null }
                sentAtSeconds = [Math]::Round($clock.Elapsed.TotalSeconds, 3)
            })
            $nextEvent++
        }

        foreach ($heldKey in $heldKeys) {
            Send-KeyDown -Window $window -VirtualKey $heldKey
        }

        $framePath = Join-Path $frameDirectory ("frame-{0:D4}.png" -f ($frameIndex + 1))
        Save-WindowFrame -Window $window -Path $framePath
    }
} finally {
    Send-KeyUp -Window $window -VirtualKey ([byte]0x57)
}

$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$videoPath = Join-Path $sessionDirectory "microduck-interactive-acceptance.mp4"
& $ffmpeg -y -hide_banner -loglevel error `
    -framerate $FramesPerSecond `
    -i (Join-Path $frameDirectory "frame-%04d.png") `
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" `
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart $videoPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $videoPath -PathType Leaf)) {
    throw "ffmpeg did not produce the visual acceptance recording."
}

$checkpointSpecs = @(
    [pscustomobject]@{ Category = "terrain"; Id = "flat_plaza"; At = 0.75 },
    [pscustomobject]@{ Category = "camera"; Id = "Side"; At = 0.75 },
    [pscustomobject]@{ Category = "camera"; Id = "Rear"; At = 1.55 },
    [pscustomobject]@{ Category = "camera"; Id = "Top"; At = 2.55 },
    [pscustomobject]@{ Category = "camera"; Id = "Showcase"; At = 3.55 },
    [pscustomobject]@{ Category = "camera"; Id = "FreeFly"; At = 5.85 },
    [pscustomobject]@{ Category = "terrain"; Id = "upstream_pyramid_stairs"; At = 8.35 },
    [pscustomobject]@{ Category = "terrain"; Id = "upstream_random_grid"; At = 11.15 },
    [pscustomobject]@{ Category = "terrain"; Id = "upstream_pyramid_slope"; At = 13.35 },
    [pscustomobject]@{ Category = "terrain"; Id = "upstream_roller_slope"; At = 15.95 },
    [pscustomobject]@{ Category = "terrain"; Id = "rock_steps"; At = 18.65 },
    [pscustomobject]@{ Category = "terrain"; Id = "stairs_bridge"; At = 21.25 }
)
$terrainCheckpoints = [ordered]@{}
$cameraCheckpoints = [ordered]@{}
foreach ($checkpoint in $checkpointSpecs) {
    $frameNumber = [Math]::Min(
        $frameCount,
        [Math]::Max(1, [int][Math]::Floor($checkpoint.At * $FramesPerSecond) + 1))
    $source = Join-Path $frameDirectory ("frame-{0:D4}.png" -f $frameNumber)
    $destination = Join-Path $sessionDirectory (
        "{0}-{1}.png" -f $checkpoint.Category, $checkpoint.Id)
    Copy-Item -LiteralPath $source -Destination $destination -Force
    if ($checkpoint.Category -eq "terrain") {
        $terrainCheckpoints[$checkpoint.Id] = $destination
    } else {
        $cameraCheckpoints[$checkpoint.Id] = $destination
    }
}

$player.Refresh()
$resolvedPlayerPath = $PlayerPath
if (-not [string]::IsNullOrWhiteSpace($player.Path)) {
    $resolvedPlayerPath = $player.Path
}
if (-not (Test-Path -LiteralPath $resolvedPlayerPath -PathType Leaf)) {
    throw "Could not resolve the captured Player executable: $resolvedPlayerPath"
}
if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) {
    throw "Player log was not produced: $logPath"
}
Start-Sleep -Milliseconds 500
$seriousLogPattern = "NullReferenceException|MissingReferenceException|DllNotFoundException|EntryPointNotFoundException|Crash!!!|MICRODUCK_PLAYER_SMOKE_FAIL|Assertion failed"
$seriousLogMatches = @(Select-String -LiteralPath $logPath -Pattern $seriousLogPattern -AllMatches -ErrorAction SilentlyContinue)

$reportPath = Join-Path $sessionDirectory "report.json"
$report = [ordered]@{
    schemaVersion = 2
    captureCompleted = $true
    requiresVisualReview = $true
    capturedAt = [DateTimeOffset]::UtcNow.ToString("o")
    playerProcessId = $player.Id
    playerPath = [System.IO.Path]::GetFullPath($resolvedPlayerPath)
    playerSha256 = (Get-FileHash -LiteralPath $resolvedPlayerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    playerLog = [System.IO.Path]::GetFullPath($logPath)
    playerLogErrorCount = $seriousLogMatches.Count
    capturedWindowTitle = $player.MainWindowTitle
    framesPerSecond = $FramesPerSecond
    frameCount = $frameCount
    durationSeconds = $durationSeconds
    events = @($eventEvidence)
    video = $videoPath
    terrainCheckpoints = $terrainCheckpoints
    cameraCheckpoints = $cameraCheckpoints
}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8

$validationPath = Join-Path $sessionDirectory "validation.json"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}
& $python (Join-Path $ProjectRoot "scripts\validate-environment-acceptance.py") `
    --input $reportPath --output $validationPath
if ($LASTEXITCODE -ne 0) {
    throw "Environment acceptance evidence validation failed: $validationPath"
}
$latestReportPath = Join-Path $OutputDirectory "latest-report.json"
$latestValidationPath = Join-Path $OutputDirectory "latest-validation.json"
Copy-Item -LiteralPath $reportPath -Destination $latestReportPath -Force
Copy-Item -LiteralPath $validationPath -Destination $latestValidationPath -Force

if ($launchedHere -and -not $KeepPlayerOpen) {
    Stop-Process -Id $player.Id -Force
}

$report | ConvertTo-Json -Depth 6 -Compress
