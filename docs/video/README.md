# Presentation video

**Final video: `AFML_live_demo.mp4`** (4:59, 1080p, subtitles burned in, no audio). It is a real screen recording of
`live_demo.py` running on our laptop, edited by `edit_recording.py`: a title card, subtitles explaining each step,
the opened figures shown full-size, and the 26-second start-up wait in step 6 played at 6x (labelled on screen).
Video files are not in git (they are large); attach the MP4 to the submission directly.

| File | What it is |
|---|---|
| `live_demo.py` | The demo: runs the project step by step (tests, dataset, MLP, live fit, mini benchmark, results), pausing for Enter |
| `run_live_demo.bat` | Opens the demo in a maximised PowerShell window |
| `RECORDING_GUIDE.md` | How to record it with Snipping Tool or Game Bar |
| `edit_recording.py` | Turns the raw recording into the final captioned video (needs ffmpeg) |

Rebuild the video from a recording:
```
python docs/video/edit_recording.py "<screen recording>.mp4" --ffmpeg <path to ffmpeg.exe>
```
The step timings in `edit_recording.py` were measured from our recording; a new recording needs them re-measured.
