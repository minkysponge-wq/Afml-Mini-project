# Recording the live demo

The demo runs the real project in a terminal, step by step, and waits for **Enter** before each step so you can
explain what is on screen. A full run takes about 6–10 minutes, depending on how long you talk; the computing
itself takes about 1.5 minutes.

## Before you record
1. Plug the laptop in and set the power mode to **Best performance** (on battery the mini benchmark crawls).
2. Close other windows and turn on *Do not disturb* so no notifications appear.
3. The dataset must exist in `data/phaseB_v2/` (it does on this laptop). On another machine, build it first:
   `python -m blinit.experiments.build_dataset --config configs/phaseB_v2.json`
4. Optional: do one practice run.

## Record
1. Double-click `docs\video\run_live_demo.bat`. A maximised PowerShell window opens with the title screen.
   Press **Ctrl +** a few times to make the text larger.
2. Start recording:
   - **Snipping Tool**: press **Win + Shift + R**, drag over the whole screen, switch the microphone on, then click **Start**.
   - or **Xbox Game Bar**: press **Win + Alt + R** (records the active window; press **Win + Alt + M** to toggle the mic).
3. Introduce the team on the title screen, then press **Enter** for each step.
4. Steps 3, 7 and 8 open a figure in the Photos app. Talk over it, then close it (Alt + F4) and click back into
   the terminal before pressing Enter.
5. At the end, stop recording: the Snipping Tool stop button, or **Win + Alt + R** again. Snipping Tool saves to
   `Videos\Screen Recordings`; Game Bar saves to `Videos\Captures`.

## The steps
| Step | Shows | Runs live? |
|---|---|---|
| 1 | Repository layout | — |
| 2 | Test suite: 29 tests pass | yes, ~20 s |
| 3 | Dataset: 3 splits, 9 variants, 9 integrity checks; example flows | yes |
| 4 | The MLP's first guesses on test cases | yes |
| 5 | One test case fitted 3 ways, every simulator run printed | yes, ~10 s |
| 6 | Mini benchmark: 18 test cases in parallel | yes, ~35 s |
| 7 | Pre-registered 450-case results and the verdict | reads result files |
| 8 | Robustness and the Cramér–Rao bound | reads result files |

Step 6 is a small live sample; say that the conclusion comes from the 450-case pre-registered test in step 7.
