# Export the slides as a silent MP4 with fixed per-slide timings (for the team to record a voice-over on).
# Usage (PowerShell, PowerPoint installed):
#   powershell -ExecutionPolicy Bypass -File docs\slides\make_video.ps1
# Output: docs\slides\AFML_project_video_silent.mp4 (1080p, 30 fps). Timings match docs\slides\video_script.md.
param(
    [string]$In = (Join-Path $PSScriptRoot "AFML_project_slides.pptx"),
    [string]$Out = (Join-Path $PSScriptRoot "AFML_project_video_silent.mp4")
)
$seconds = @(15, 40, 40, 40, 40, 75, 30, 35, 45, 40, 25)   # slides 1..11, total 7 min 5 s

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($In, $true, $false, $false)   # read-only: the .pptx on disk is not changed
if ($pres.Slides.Count -ne $seconds.Count) { $pres.Close(); $ppt.Quit(); throw "expected $($seconds.Count) slides, found $($pres.Slides.Count)" }
for ($i = 1; $i -le $pres.Slides.Count; $i++) {
    $t = $pres.Slides.Item($i).SlideShowTransition
    $t.AdvanceOnTime = -1          # msoTrue
    $t.AdvanceTime = $seconds[$i - 1]
}
if (Test-Path $Out) { Remove-Item $Out }
# CreateVideo(FileName, UseTimingsAndNarrations, DefaultSlideDuration, VertResolution, FramesPerSecond, Quality)
$pres.CreateVideo($Out, $true, 5, 1080, 30, 85)
while ($pres.CreateVideoStatus -eq 1 -or $pres.CreateVideoStatus -eq 2) { Start-Sleep -Seconds 2 }   # queued / in progress
$status = $pres.CreateVideoStatus
$pres.Close(); $ppt.Quit()
if ($status -ne 3) { throw "video export failed (status $status)" }                              # 3 = done
"wrote $Out"
