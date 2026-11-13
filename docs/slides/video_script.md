# Presentation video: timings and narration draft

The silent video (`AFML_project_video_silent.mp4`, made by `make_video.ps1`) holds each slide for the time below.
Record your voice over it: in PowerPoint, *Slide Show › Record*, or any screen recorder or video editor.

**This narration is a draft written with an AI assistant.** The course asks for your own words, so use it only as a
checklist of points and say it your own way. Every number below is on the slide it belongs to.

| Slide | Time | Start at | Who | Points to cover |
|---|---|---|---|---|
| 1 Title | 15 s | 0:00 | [name] | Project title; each member says their name. |
| 2 Topic | 40 s | 0:15 | [name] | We estimate a fluid's viscosity from a few noisy sensor readings. The usual way reruns a simulator until it matches, and each run costs time. Question: does a neural network's first guess cut the number of runs? The test is fair: against the best classical search, counting training cost. |
| 3 Uniqueness | 40 s | 0:55 | [name] | Strong baseline (Brent), not just a bad starting guess. Training cost counted, so we give a break-even. Test rules written down before the test. Accuracy checked against the Cramér–Rao limit. The negative result is reported honestly. |
| 4 Dataset | 40 s | 1:35 | [name] | Our own solver, PDEBench scheme, no public dataset. 3,000 / 50 / 50 trajectories, 9 sensor/noise variants each. Split by trajectory seed; 9 integrity checks. Three extra robustness sets. |
| 5 Design | 40 s | 2:15 | [name] | Walk through the diagram: sensors → MLP guess → refiner → estimate. Every method shares the same simulator, bounds and run counter. |
| 6 Results | 75 s | 2:55 | [name] | All methods are equally accurate (0.22% median), at the Cramér–Rao limit. The learned start saves ~2 CPU-s vs a midpoint start, 0.8–1.1 vs Brent at the loose tolerance, ~0 at the tight one. Pre-registered verdict: not positive, 2× not met. Follow-up bracket: 8–10%. Break-even 2,600–4,500 queries; the saving disappears on unseen inputs. |
| 7 Done vs remaining | 30 s | 4:10 | [name] | Everything is done except an optional larger test set; this video completes the list. |
| 8 Quantity of work | 35 s | 4:40 | [name] | All 9 components complete and running; 29 tests pass. Mention the slow parts (fine-grid build, 45 min per test set). |
| 9 Learning points | 45 s | 5:15 | [name] | Pick the strongest baseline; compare at matched precision; count everything; pre-registration kept us honest; a network helps only on inputs like its training data. |
| 10 Challenges | 40 s | 6:00 | [name] | 1-D search is already cheap; synthetic data only; cost model is steps × fixed time; small test sets; failure on unseen inputs. |
| 11 References | 25 s | 6:40 | [name] | Name the two closest papers (Sambharya et al.; Zhou et al.) and PDEBench. |
| End | — | 7:05 | | |

To change a slide's time, edit `$seconds` in `make_video.ps1` (one entry per slide) and run it again.
