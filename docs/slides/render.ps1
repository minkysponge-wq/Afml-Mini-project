param([string]$In, [string]$OutDir)
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($In, $true, $false, $false)
New-Item -ItemType Directory -Force $OutDir | Out-Null
$i = 1
foreach ($s in $pres.Slides) { $s.Export((Join-Path $OutDir ("slide{0:D2}.png" -f $i)), "PNG", 1280, 720); $i++ }
$pres.Close(); $ppt.Quit()
