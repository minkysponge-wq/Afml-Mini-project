"""Fill the course slide template with this project's content.

Usage: python docs/slides/build_slides.py
Input:  docs/slides/template.pptx (course template, unchanged)
Output: docs/slides/AFML_project_slides.pptx
Figures come from docs/report/figures (built by docs/report/make_figures.py); numbers are from results/ (see
RESEARCH_LOG.md). Team names and SRNs are left for the team to fill in.
"""
from __future__ import annotations

import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
LOGO = "Picture 12"


def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def drop(slide, name):
    sh = shape(slide, name)
    sh._element.getparent().remove(sh._element)


def set_bullets(sh, items, size=None):
    """Replace the text, reusing the first paragraph's formatting (bullets, colour, font) for every item.
    An item may be a string or a list of (text, bold) pieces."""
    tf = sh.text_frame
    first = tf.paragraphs[0]._p
    proto_p = copy.deepcopy(first)
    proto_r = copy.deepcopy(first.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r")[0])
    for p in list(tf.paragraphs):
        p._p.getparent().remove(p._p)
    body = tf._txBody
    for item in items:
        p = copy.deepcopy(proto_p)
        for r in p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r"):
            p.remove(r)
        for br in p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}br"):
            p.remove(br)
        pieces = [(item, False)] if isinstance(item, str) else item
        ppr = p.find("{http://schemas.openxmlformats.org/drawingml/2006/main}pPr")
        if ppr is not None:
            ppr.set("algn", "l")        # the template justifies; justified short lines leave wide gaps
        end = p.find("{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr")
        for text, bold in pieces:
            r = copy.deepcopy(proto_r)
            r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = text
            rpr = r.find("{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
            if rpr is not None:
                rpr.set("b", "1" if bold else "0")
                if size:
                    rpr.set("sz", str(int(size * 100)))
            if end is not None:
                end.addprevious(r)
            else:
                p.append(r)
        body.append(p)


def fill_table(sh, rows, size=12, start=1):
    t = sh.table
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = t.cell(start + i, j)
            tf = cell.text_frame
            for p in list(tf.paragraphs)[1:]:
                p._p.getparent().remove(p._p)
            para = tf.paragraphs[0]
            for r in list(para.runs):
                r._r.getparent().remove(r._r)
            run = para.add_run()
            run.text = text
            run.font.size = Pt(size)
            run.font.name = "Trebuchet MS"


def picture(slide, path, left, top, width=None, height=None):
    kw = {}
    if width is not None:
        kw["width"] = Inches(width)
    if height is not None:
        kw["height"] = Inches(height)
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


def main():
    prs = Presentation(HERE / "template.pptx")
    S = prs.slides

    # 1 Title
    s = S[0]
    set_bullets(shape(s, "Google Shape;26;p3"), [
        [("Project Title: ", True), ("Does a Learned Starting Point Make Viscosity Estimation Cheaper?", False)],
        [("A cost-counted study on the Burgers equation", False)],
        "", "",
        "Team member 1 SRN and name:",
        "Team member 2 SRN and name:",
        "Team member 3 SRN and name:",
        "Team member 4 SRN and name:",
    ])

    # 2 Problem statement
    s = S[1]
    set_bullets(shape(s, "Content Placeholder 2"), [
        [("Goal: ", True), ("find the viscosity of a fluid (1-D Burgers equation) from a few noisy sensor readings", False)],
        [("Standard way: ", True), ("run the simulator again and again until it matches the readings; each run costs time", False)],
        [("Our question: ", True), ("if a small neural network guesses the viscosity first, does the fit need fewer runs?", False)],
        [("Fair test: ", True), ("against the best classical search, counting every run and the network's training cost", False)],
    ], size=18)
    drop(s, "Picture 2")
    sh = shape(s, "Content Placeholder 2")
    sh.width, sh.height = Inches(11.0), Inches(2.6)
    picture(s, FIG / "trajectories.png", 2.4, 4.45, width=8.2)

    # 3 Uniqueness
    s = S[2]
    set_bullets(shape(s, "Content Placeholder 2"), [
        "Compared with a strong baseline (Brent's search), not only with a poor starting guess",
        "Counts the network's training cost and gives a break-even number of queries",
        "Test protocol written down and committed before the test data were used",
        "Checks accuracy against the theoretical limit (Cramér–Rao bound)",
        "Reports a negative result honestly: the 2x target was not met",
    ], size=18)
    shape(s, "Content Placeholder 2").width = Inches(6.0)
    drop(s, "Picture 6")
    picture(s, FIG / "test_paired.png", 7.15, 2.35, width=5.55)

    # 4 Dataset
    s = S[3]
    set_bullets(shape(s, "Content Placeholder 2"), [
        [("Source: ", True), ("generated by our own Burgers solver (no public dataset; PDEBench scheme)", False)],
        [("Size: ", True), ("3,000 train / 50 val / 50 test trajectories = 27,000 / 450 / 450 cases", False)],
        [("Attributes: ", True), ("16 sensors x 4 times; 4, 8 or 16 sensors on; 0 / 1 / 5% noise; known starting state", False)],
        [("Target: ", True), ("log viscosity, viscosity in [0.01, 0.1]", False)],
        [("Robustness sets: ", True), ("3 x 50 new trajectories (new starting states, finer grid, unseen sensor patterns)", False)],
        [("Checks: ", True), ("split by trajectory seed; 9 automated integrity checks", False)],
    ], size=16)
    shape(s, "Content Placeholder 2").width = Inches(6.2)
    drop(s, "Picture 2")
    picture(s, FIG / "objective_curves.png", 7.65, 2.2, width=5.2)

    # 5 Design diagram
    s = S[4]
    drop(s, "Picture 10")
    sh = shape(s, "Content Placeholder 2")
    set_bullets(sh, [
        "Same simulator, bounds and cost accounting for every method; every simulator run is counted",
    ], size=14)
    sh.top, sh.height = Inches(6.25), Inches(0.5)
    picture(s, FIG / "pipeline.png", 0.9, 1.9, width=11.2)

    # 6 Results
    s = S[5]
    set_bullets(shape(s, "Content Placeholder 2"), [
        "Same accuracy for all methods (0.22% median error); matches the Cramér–Rao limit",
        "Learned start vs midpoint start: about 2 CPU-s saved per query on the pilot test",
        "Learned start vs Brent: 0.8–1.1 CPU-s saved at the loose tolerance, ~0 at the tight one",
        "Pre-registered test: not positive; 2x target not met (1.2x at matched typical precision)",
        "Break-even 2,600–4,500 queries; saving gone for unseen starting states / sensor patterns",
    ], size=14)
    shape(s, "Content Placeholder 2").width = Inches(5.5)
    drop(s, "Picture 10")
    picture(s, FIG / "tolerance_curves.png", 6.75, 2.0, width=6.1)
    picture(s, FIG / "robustness.png", 6.75, 4.3, width=6.1)

    # 7 Done vs remaining
    t7 = shape(S[6], "Table 6")
    for r in list(t7.table.rows)[1:]:
        r.height = Inches(0.5)          # template rows run into the footer
    fill_table(t7, [
        ("1", "Burgers solver + 11 correctness tests (exact Cole–Hopf solution)", "Done"),
        ("2", "Classical methods: Brent, midpoint / grid / kNN + least squares", "Done"),
        ("3", "Dataset generator, 9 integrity checks; MLP trained (3 seeds)", "Done"),
        ("4", "Pre-registered test evaluation and go/no-go verdict", "Done"),
        ("5", "Tolerance curves, Cramér–Rao bound, robustness sets, ablations", "Done"),
        ("6", "IEEE paper, slides, repository documentation", "Done"),
        ("7", "Larger 200-trajectory test set (optional); presentation video", "To be done"),
    ], size=13)

    # 8 Quantity and quality of work
    fill_table(shape(S[7], "Content Placeholder 3"), [
        ("1", "Burgers finite-volume solver", "100", "Y", "~1 s per run at high viscosity"),
        ("2", "Inverse methods (Brent, LSQ, starts)", "100", "Y", "—"),
        ("3", "Dataset build + integrity checks", "100", "Y", "~4 min on 8 cores"),
        ("4", "MLP training and tuning", "100", "Y", "—"),
        ("5", "Paired evaluation (resumable)", "100", "Y", "~45 min per test set, 12 workers"),
        ("6", "Bootstrap analysis + break-even", "100", "Y", "—"),
        ("7", "Tolerance curves / Cramér–Rao bound", "100", "Y", "—"),
        ("8", "Robustness sets + ablations", "100", "Y", "fine-grid build is slowest"),
        ("9", "Test suite (pytest)", "100", "Y", "26 tests pass"),
    ], size=11)

    # 9 Learning points
    fill_table(shape(S[8], "Content Placeholder 7"), [
        ("1", "Pick the strongest baseline: a learned start that beats a midpoint guess can still lose to a direct search"),
        ("2", "A speed-up can depend on the stopping rule: compare at matched precision, not at one tolerance pair"),
        ("3", "Count everything: derivative runs and training cost change the verdict (break-even in queries)"),
        ("4", "Pre-registering the test stopped us from tuning to the result we hoped for"),
        ("5", "A network helps only on inputs like its training data; refinement hides the error but not the cost"),
    ], size=12)

    # 10 Unresolved challenges
    fill_table(shape(S[9], "Content Placeholder 7"), [
        ("1", "One equation, one scalar unknown: a 1-D search is already cheap", "Scope"),
        ("2", "Synthetic data only; the finer-grid set is the only model-mismatch test", "Data"),
        ("3", "Cost = solver steps x fixed time per step (laptop timings not clean)", "Design"),
        ("4", "Small test sets (50 trajectories) give wide confidence intervals", "Data"),
        ("5", "Learned start fails on unseen sensor patterns and starting states", "Design"),
    ], size=12)

    # 11 References
    fill_table(shape(S[10], "Content Placeholder 7"), [
        ("1", "Learning to Warm-Start Fixed-Point Optimization Algorithms", "R. Sambharya, G. Hall, B. Amos, B. Stellato", "2024"),
        ("2", "A Neural Network Warm-Start Approach for the Inverse Acoustic Obstacle Scattering Problem", "M. Zhou, J. Han, M. Rachh, C. Borges", "2023"),
        ("3", "PDEBench: An Extensive Benchmark for Scientific Machine Learning", "M. Takamoto et al.", "2022"),
        ("4", "Tutorial on Amortized Optimization", "B. Amos", "2023"),
        ("5", "Deep Learning for Fast Inference of Mechanistic Models' Parameters", "M. Borisyak, S. Born, P. Neubauer, M. N. Cruz-Bournazou", "2023"),
        ("6", "The Frontier of Simulation-Based Inference", "K. Cranmer, J. Brehmer, G. Louppe", "2020"),
        ("7", "A Sparse Control Approach to Optimal Sensor Placement in PDE-Constrained Parameter Estimation", "I. Neitzel, K. Pieper, B. Vexler, D. Walter", "2019"),
    ], size=10)

    out = HERE / "AFML_project_slides.pptx"
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
