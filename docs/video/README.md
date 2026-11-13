# Presentation video

**Final video: `AFML_live_demo.mp4`** (4:59, 1080p, subtitles burned in, no audio). It is a real screen recording of
`live_demo.py` running on our laptop, edited by `edit_recording.py`: a title card, subtitles explaining each step,
the opened figures shown full-size, and the 26-second start-up wait in step 6 played at 6x (labelled on screen).
Rebuild: `python docs/video/edit_recording.py "<screen recording>.mp4" --ffmpeg <path to ffmpeg.exe>`.

The rest of this file describes an earlier, generated (non-recorded) video.


`AFML_project_video.mp4` (10:43, 1080p, captioned, no audio) covers the problem, the datasets, a live demo, the
benchmarks and the results. It is not in git (about 80 MB); rebuild it with the steps below.

| File | What it is |
|---|---|
| `demo_run.py` | The live demo: loads the dataset, then solves test case 4 with Brent, MLP+LSQ and the MLP bracket, printing every simulator call |
| `demo_output.txt`, `demo_trace.json` | Its captured output (matches `results/*_test/runs.csv` for case 4 exactly) |
| `pytest_output.txt` | Captured `python -m pytest -v` output (29 passed) |
| `build_video.py` | Draws the 37 frames (1920×1080) from saved results and the captures, then builds `AFML_video.pptx` |
| `export_video.ps1` | Exports the deck to MP4 through PowerPoint, holding each frame for its time in `timings.json` |
| `pes_logo.png` | Logo taken from the course slide template |

Case 4 was chosen by a fixed rule (first test trajectory, 8 sensors, 1% noise), not for its result. On that case the
learned methods look better than they do on average, so the video says "one case only" and shows the 450-case
results afterwards.

## Rebuild
```
python -m blinit.experiments.build_dataset --config configs/phaseB_v2.json --workers 8
git checkout -- data/phaseB_v2/integrity.json
python docs/video/demo_run.py --variant 4 --json docs/video/demo_trace.json > docs/video/demo_output.txt
python docs/video/build_video.py
powershell -ExecutionPolicy Bypass -File docs/video/export_video.ps1
```
`pytest_output.txt` was captured with `python -m pytest -v`, keeping only the PASSED lines and the summary line.

The video has captions but no voice-over. If the course needs narration, record it over the MP4, e.g. with
PowerPoint's *Record* on `AFML_video.pptx` or any video editor.
