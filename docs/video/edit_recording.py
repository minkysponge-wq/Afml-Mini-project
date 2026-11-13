"""Turn the raw screen recording of live_demo.py into the final captioned video.

Usage: python docs/video/edit_recording.py "<recording.mp4>" [--ffmpeg path/to/ffmpeg.exe]
Output: docs/video/AFML_live_demo.mp4 (1920x1080, 30 fps, subtitles burned in, no audio).

Edit decisions (times are in the raw recording, measured from its frames):
- 5 s title card added at the start.
- While a figure window was open, the same figure file is shown full-size (the window was moved around on screen).
- After 311.4 s the PowerShell prompt (showing a local folder path) appears; the video ends just before it,
  holding the last frame for 4 s.
- 151.5-177.5 s: step 6 start-up wait (worker processes starting, nothing on screen changes) is played at 6x and
  labelled as sped up. Nothing else is cut or sped up.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "edit_build"
W, H, BAR = 1920, 1080, 170          # output size; subtitle bar height at the bottom
TOPH = H - BAR                        # 910: area for the recording
BG, BARBG = "#0C0C0C", "#151B27"
INTRO = 5.0
SP_A, SP_B, SPEED = 151.5, 177.5, 6   # sped-up interval in the raw recording
FIGS = [(62.0, 84.3, ROOT / "docs/report/figures/trajectories.png", "docs/report/figures/trajectories.png"),
        (212.1, 238.0, ROOT / "results/report/figures/test_paired_vs_brent.png", "results/report/figures/test_paired_vs_brent.png"),
        (270.3, 302.7, ROOT / "docs/report/figures/robustness.png", "docs/report/figures/robustness.png")]
END_SRC = 311.4                      # the PowerShell prompt (with a local path) appears at 311.5 s; end before it
END_HOLD = 4.0

# Subtitles, in raw-recording seconds (start, end, text). Numbers match what is on screen / in the paper.
CUES = [
    (0.0, 10.0, "Live demo of our AFML project: estimating the viscosity of a fluid (1-D Burgers equation)"
                " from a few noisy sensor readings. Everything shown runs live from our repository."),
    (10.0, 20.0, "Step 1 - The repository: our own Burgers solver, the inverse methods (Brent search, least squares,"
                 " learned bracket), the dataset builder and the MLP."),
    (20.0, 30.0, "Every experiment is a versioned config file, the test rules were written down before testing"
                 " (pre-registration), and every number in the paper comes from a results file."),
    (30.0, 42.0, "Step 2 - Running the test suite live: the solver is checked against an exact (Cole-Hopf) solution,"
                 " mass conservation and symmetry."),
    (42.0, 53.0, "The suite also checks the dataset for leaks between training, validation and test data,"
                 " and checks the model code."),
    (53.0, 60.0, "Result: all 29 tests pass."),
    (60.0, 67.0, "Step 3 - The dataset. No public dataset fits this question, so we generated our own with the"
                 " solver. Here are example flows."),
    (67.0, 75.5, "Low viscosity keeps sharp fronts; high viscosity smooths the wave quickly."
                 " The dots are the 16 sensor readings."),
    (75.5, 84.3, "Unknown: the viscosity (0.01 to 0.1). Known: the starting wave. Observed: velocity at 4, 8 or 16"
                 " sensors at 4 times, with 0, 1 or 5% noise."),
    (84.3, 91.0, "3,000 / 50 / 50 trajectories, each observed 9 ways (27,000 / 450 / 450 cases)."
                 " Split by seed; all 9 integrity checks pass."),
    (91.0, 103.0, "Step 4 - The trained MLP (2 hidden layers of 64) reads the sensor readings and the known starting"
                  " wave, and guesses the viscosity."),
    (103.0, 113.0, "It answers in a fraction of a millisecond (0.13 ms for one case). One simulator run takes 0.1 to 1 s."),
    (113.0, 121.0, "Median error of its guess on the 450 test cases: 10.2%. Good, but far from the 0.22% the data"
                   " allow, so a classical method must refine it."),
    (121.0, 129.0, "Step 5 - One test case (8 sensors, 1% noise) solved live three ways. Each line is one real"
                   " simulator run, and its cost is counted."),
    (129.0, 137.0, "Brent's search needs no guess: 10 runs, 4.16 CPU-seconds."
                   " MLP guess + least squares: 10 runs, 2.99 CPU-seconds."),
    (137.0, 144.0, "MLP guess used as a search bracket for Brent: 7 runs, 2.17 CPU-seconds."),
    (144.0, 151.5, "All three find the same viscosity (0.01194; true value 0.01186). They differ only in cost,"
                   " and this is just one case."),
    (151.5, 183.0, "Step 6 - Mini benchmark: the same three methods on 18 test cases (2 per sensor/noise setting),"
                   " run live in parallel. The start-up wait is sped up 6x."),
    (183.0, 196.0, "Each row is one test case: its cost in CPU-seconds for Brent, MLP + least squares,"
                   " and the MLP bracket."),
    (196.0, 211.0, "Average on these 18 cases: Brent 9.29, MLP + least squares 7.54, bracket 7.80 CPU-seconds."
                   " 18 cases are too few to conclude anything; the full test comes next."),
    (211.0, 225.0, "Step 7 - The pre-registered test: 450 test cases, 3 MLP seeds, success rules fixed before"
                   " the test was run. Each line: MLP + least squares minus Brent, with a 95% interval."),
    (225.0, 238.0, "Loose stopping rule (left panel): the MLP route saves 0.8 to 1.1 CPU-seconds per query."
                   " Accuracy-matched stopping (right panel): every interval crosses zero."),
    (238.0, 249.0, "All methods are equally accurate (median error 0.22%), so only the cost differs. The learned"
                   " guess makes least squares about 2 CPU-seconds cheaper than starting in the middle."),
    (249.0, 260.0, "Verdict: the learned start does not reliably beat Brent's search."
                   " The pre-registered result is not positive and the 2x target is not met."),
    (260.0, 270.3, "Follow-up: using the MLP guess as a bracket for Brent saves about 1 CPU-second per query"
                   " (8-10%). It was designed after the main test, so it is weaker evidence."),
    (270.3, 281.0, "Step 8 - Robustness on three new test sets. Left: MLP + least squares minus Brent."
                   " Middle: error of the MLP alone and after refinement. Right: cost per query."),
    (281.0, 292.0, "Higher-frequency starting waves and unseen sensor patterns: the MLP's guess gets worse and"
                   " the saving disappears. Data from a finer grid: the saving holds."),
    (292.0, 302.7, "Refinement always recovers full accuracy, and both methods reach the Cramer-Rao bound:"
                   " no method can be more accurate on these data."),
    (302.7, 999.0, "Conclusion: a learned guess helps least squares, but on this one-parameter problem Brent's"
                   " search is hard to beat. Code, data and paper: github.com/minkysponge-wq/Afml-Mini-project"),
]


def out_time(src):
    """Map a raw-recording time to the edited video's time."""
    if src <= SP_A:
        return src + INTRO
    if src <= SP_B:
        return INTRO + SP_A + (src - SP_A) / SPEED
    return src + INTRO - (SP_B - SP_A) + (SP_B - SP_A) / SPEED


def ts(t):
    h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def make_stills():
    BUILD.mkdir(exist_ok=True)
    # title card
    fig = plt.figure(figsize=(W / 100, TOPH / 100), dpi=100, facecolor=BG)
    fig.text(0.5, 0.80, "AFML Course Project  -  Live Demo", ha="center", fontsize=30, color="#2BB5C0", family="Segoe UI", weight="bold")
    fig.text(0.5, 0.60, "Does a Learned Starting Point Make\nViscosity Estimation Cheaper?", ha="center", va="center",
             fontsize=50, color="#FFFFFF", family="Segoe UI", weight="bold", linespacing=1.2)
    fig.text(0.5, 0.43, "A cost-counted study on the Burgers equation  |  UE24AM342AA1, PES University",
             ha="center", fontsize=24, color="#C9D4E3", family="Segoe UI")
    team = ["PES1UG24AM246  Sampreeth Jaladi", "PES1UG24AM243  Samarth", "PES1UG24AM255  Sarva Sandip",
            "PES1UG24AM239  Sai Likhith S"]
    for k, t in enumerate(team):
        fig.text(0.5, 0.31 - k * 0.055, t, ha="center", fontsize=24, color="#E6E6E6", family="Segoe UI")
    fig.savefig(BUILD / "intro.png", dpi=100, facecolor=BG); plt.close(fig)
    # figure stills: the file the demo opened, fitted on a white card
    for i, (_, _, path, label) in enumerate(FIGS, 1):
        img = mpimg.imread(path); ih, iw = img.shape[:2]
        fig = plt.figure(figsize=(W / 100, TOPH / 100), dpi=100, facecolor=BG)
        maxw, maxh = 1840, TOPH - 90          # keep the top 50 px free for the label
        s = min(maxw / iw, maxh / ih); w, h = iw * s, ih * s
        x0, y0 = (W - w) / 2, (TOPH - 50 - h) / 2
        ax = fig.add_axes([x0 / W, y0 / TOPH, w / W, h / TOPH]); ax.imshow(img); ax.axis("off")
        fig.text(0.012, 0.975, f"Figure opened by the demo: {label}", fontsize=17, color="#9DA5B4", family="Consolas", va="top")
        fig.savefig(BUILD / f"fig{i}.png", dpi=100, facecolor=BG); plt.close(fig)


def make_ass():
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
             "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic,"
             " Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL,"
             " MarginR, MarginV, Encoding",
             "Style: Sub,Segoe UI,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,90,90,38,1",
             "Style: Tag,Segoe UI,30,&H0000D7FF,&H0000D7FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,9,40,40,30,1",
             "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    end = out_time(END_SRC) + END_HOLD
    lines.append(f"Dialogue: 0,{ts(0)},{ts(INTRO)},Sub,,0,0,0,,Recorded live on our laptop by running docs/video/live_demo.py")
    for a, b, text in CUES:
        lines.append(f"Dialogue: 0,{ts(out_time(a))},{ts(min(out_time(b), end))},Sub,,0,0,0,,{text}")
    lines.append(f"Dialogue: 1,{ts(out_time(SP_A))},{ts(out_time(SP_B))},Tag,,0,0,0,,>> 6x speed (worker processes starting)")
    (BUILD / "subs.ass").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = ap.parse_args()
    make_stills()
    total = make_ass()
    rec = str(Path(a.recording).resolve())
    norm = f"fps=30,format=yuv420p,setsar=1"
    base = f"crop=1916:1104:0:38,scale=1580:910:flags=lanczos,pad={W}:{TOPH}:170:0:color={BG[1:]}"
    segs, inputs, k = [], ["-i", rec, "-loop", "1", "-t", str(INTRO), "-framerate", "30", "-i", str(BUILD / "intro.png")], 2
    filt = [f"[1:v]{norm}[intro]"]; order = ["[intro]"]

    def rec_seg(a0, b0, name, extra=""):
        filt.append(f"[0:v]trim={a0}:{b0},setpts=PTS-STARTPTS{extra},{base},{norm}[{name}]"); order.append(f"[{name}]")

    rec_seg(0, FIGS[0][0], "s1")
    for i, (fa, fb, _, _) in enumerate(FIGS, 1):
        inputs += ["-loop", "1", "-t", f"{fb - fa:.3f}", "-framerate", "30", "-i", str(BUILD / f"fig{i}.png")]
        filt.append(f"[{k}:v]{norm}[f{i}]"); k += 1
    order_fig = {1: "[f1]", 2: "[f2]", 3: "[f3]"}
    order.append(order_fig[1])
    rec_seg(FIGS[0][1], SP_A, "s2")
    rec_seg(SP_A, SP_B, "sp", f",setpts=PTS/{SPEED}")
    rec_seg(SP_B, FIGS[1][0], "s3")
    order.append(order_fig[2])
    rec_seg(FIGS[1][1], FIGS[2][0], "s4")
    order.append(order_fig[3])
    rec_seg(FIGS[2][1], END_SRC, "s5", "")
    filt[-1] = filt[-1].replace("[s5]", f",tpad=stop_mode=clone:stop_duration={END_HOLD}[s5]")
    filt.append("".join(order) + f"concat=n={len(order)}:v=1:a=0[cat]")
    filt.append(f"[cat]pad={W}:{H}:0:0:color={BARBG[1:]},subtitles=subs.ass[out]")
    cmd = [a.ffmpeg, "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
           str(HERE / "AFML_live_demo.mp4")]
    subprocess.run(cmd, cwd=BUILD, check=True)
    print(f"wrote {HERE / 'AFML_live_demo.mp4'} (expected length {total:.1f} s)")


if __name__ == "__main__":
    main()
