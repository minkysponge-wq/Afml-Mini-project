# Export docs/video/AFML_video.pptx (built by build_video.py) to an MP4, holding each frame for the time in timings.json.
# Usage: powershell -ExecutionPolicy Bypass -File docs\video\export_video.ps1
param(
    [string]$In = (Join-Path $PSScriptRoot "AFML_video.pptx"),
    [string]$Out = (Join-Path $PSScriptRoot "AFML_project_video.mp4")
)
$times = Get-Content (Join-Path $PSScriptRoot "timings.json") -Raw | ConvertFrom-Json
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($In, $true, $false, $false)
if ($pres.Slides.Count -ne $times.Count) { $pres.Close(); $ppt.Quit(); throw "slide count $($pres.Slides.Count) != timings $($times.Count)" }
for ($i = 1; $i -le $pres.Slides.Count; $i++) {
    $t = $pres.Slides.Item($i).SlideShowTransition
    $t.AdvanceOnTime = -1
    $t.AdvanceTime = [double]$times[$i - 1]
    $t.EntryEffect = 0            # no transition: frames of the demo animation must cut, not fade
}
if (Test-Path $Out) { Remove-Item $Out }
$pres.CreateVideo($Out, $true, 5, 1080, 30, 90)
while ($pres.CreateVideoStatus -eq 1 -or $pres.CreateVideoStatus -eq 2) { Start-Sleep -Seconds 2 }
$status = $pres.CreateVideoStatus
$pres.Close(); $ppt.Quit()
if ($status -ne 3) { throw "video export failed (status $status)" }
"wrote $Out"
