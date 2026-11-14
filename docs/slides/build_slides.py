"""Fill the course slide template with this project's content.

Usage: python docs/slides/build_slides.py
Input:  docs/slides/template.pptx (course template, unchanged)
Output: docs/slides/AFML_project_slides.pptx
Figures come from docs/report/figures (built by docs/report/make_figures.py); numbers are from results/ (see
RESEARCH_LOG.md). Layout rule: no picture may overlap body text or the footer (enforced by check_layout).
"""
from __future__ import annotations

import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
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


BODY = "1F2937"      # charcoal body text (replaces the template's bright #0033CC blue)
LABEL = "1F3A93"     # navy for bold labels
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def style(sh, left, top, width, height, space_after=8, line=1.1):
    """Place a text box and give its text the deck's colours and spacing (wrap on, no autofit)."""
    sh.left, sh.top, sh.width, sh.height = Inches(left), Inches(top), Inches(width), Inches(height)
    tf = sh.text_frame
    tf.word_wrap = True
    bp = tf._txBody.find(A + "bodyPr")
    for child in list(bp):
        if child.tag in (A + "normAutofit", A + "spAutoFit", A + "noAutofit"):
            bp.remove(child)
    bp.append(bp.makeelement(A + "noAutofit", {}))
    for p in tf.paragraphs:
        p.space_after = Pt(space_after)
        p.line_spacing = line
        for r in p.runs:
            r.font.color.rgb = RGBColor.from_string(LABEL if r.font.bold else BODY)


def fit(path, width):
    """Height (inches) of an image scaled to the given width."""
    w, h = Image.open(path).size
    return width * h / w


def check_layout(prs, footer_top=6.93):
    """Fail the build if a picture overlaps a body text box or runs into the footer."""
    E = 914400
    for i, s in enumerate(prs.slides, 1):
        pics = [sh for sh in s.shapes if sh.shape_type == 13 and sh.name != LOGO]
        texts = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip() and 1.8 < sh.top / E < 6.9]
        for p in pics:
            pb = (p.left / E, p.top / E, (p.left + p.width) / E, (p.top + p.height) / E)
            assert pb[3] <= footer_top and pb[2] <= 13.2, f"slide {i}: picture leaves the content area {pb}"
            for t in texts:
                tb = (t.left / E, t.top / E, (t.left + t.width) / E, (t.top + t.height) / E)
                ox = min(pb[2], tb[2]) - max(pb[0], tb[0]); oy = min(pb[3], tb[3]) - max(pb[1], tb[1])
                assert not (ox > 0.01 and oy > 0.01), f"slide {i}: picture overlaps text box '{t.name}'"


def main():
    prs = Presentation(HERE / "template.pptx")
    S = prs.slides

    # 1 Title
    s = S[0]
    sh = shape(s, "Google Shape;26;p3")
    set_bullets(sh, [
        [("Project Title: ", True), ("Does a Learned Starting Point Make Viscosity Estimation Cheaper?", True)],
        [("A cost-counted study on the Burgers equation", False)],
        "",
        "Team member 1 SRN and name: PES1UG24AM246 Sampreeth Jaladi",
        "Team member 2 SRN and name: PES1UG24AM243 Samarth",
        "Team member 3 SRN and name: PES1UG24AM255 Sarva Sandip",
        "Team member 4 SRN and name: PES1UG24AM239 Sai Likhith S",
    ], size=22)
    style(sh, 0.83, 2.1, 11.6, 4.6, space_after=6)

    # 2 Problem statement: text across the top, figure centred below
    s = S[1]
    sh = shape(s, "Content Placeholder 2")
    set_bullets(sh, [
        [("Goal: ", True), ("find the viscosity of a fluid (1-D Burgers equation) from a few noisy sensor readings", False)],
        [("Standard way: ", True), ("rerun the simulator until it matches the readings; every run costs time", False)],
        [("Our question: ", True), ("if a small neural network guesses the viscosity first, does the fit need fewer runs?", False)],
        [("Fair test: ", True), ("against the best classical search, counting every run and the network's training cost", False)],
    ], size=17)
    style(sh, 1.0, 1.95, 11.3, 1.95, space_after=6)
    drop(s, "Picture 2")
    w = 9.2
    picture(s, FIG / "trajectories.png", (13.333 - w) / 2, 4.05, width=w)

    # 3 Uniqueness: text left, figure right
    s = S[2]
    sh = shape(s, "Content Placeholder 2")
    set_bullets(sh, [
        "Compared with a strong baseline (Brent's search), not only with a poor starting guess",
        "Counts the network's training cost and gives a break-even number of queries",
        "Test protocol written down and committed before the test data were used",
        "Checks accuracy against the theoretical limit (Cramér–Rao bound)",
        "Reports a negative result honestly: the 2x target was not met",
    ], size=17)
    style(sh, 0.9, 2.0, 5.9, 4.7, space_after=10)
    drop(s, "Picture 6")
    w = 5.9
    picture(s, FIG / "test_paired.png", 7.0, 2.0 + (4.7 - fit(FIG / "test_paired.png", w)) / 2, width=w)

    # 4 Dataset: text left, figure right
    s = S[3]
    sh = shape(s, "Content Placeholder 2")
    set_bullets(sh, [
        [("Source: ", True), ("generated by our own Burgers solver (no public dataset; PDEBench scheme)", False)],
        [("Size: ", True), ("3,000 train / 50 val / 50 test trajectories = 27,000 / 450 / 450 cases", False)],
        [("Attributes: ", True), ("16 sensors x 4 times; 4, 8 or 16 sensors on; 0 / 1 / 5% noise; known starting state", False)],
        [("Target: ", True), ("log viscosity, viscosity in [0.01, 0.1]", False)],
        [("Robustness sets: ", True), ("3 x 50 new trajectories (new starting states, finer grid, unseen sensor patterns)", False)],
        [("Checks: ", True), ("split by trajectory seed; 9 automated integrity checks", False)],
    ], size=15)
    style(sh, 0.9, 1.95, 6.2, 4.85, space_after=7)
    drop(s, "Picture 2")
    w = 5.4
    picture(s, FIG / "objective_curves.png", 7.45, 1.95 + (4.85 - fit(FIG / "objective_curves.png", w)) / 2, width=w)

    # 5 Design diagram: diagram centred, one-line caption under it
    s = S[4]
    drop(s, "Picture 10")
    sh = shape(s, "Content Placeholder 2")
    set_bullets(sh, [
        "Same simulator, bounds and cost accounting for every method; every simulator run is counted",
    ], size=14)
    w = 10.0
    h = fit(FIG / "pipeline.png", w)
    picture(s, FIG / "pipeline.png", (13.333 - w) / 2, 1.9, width=w)
    style(sh, 1.2, 1.9 + h + 0.1, 10.9, 0.45, space_after=0)
    p0 = sh.text_frame.paragraphs[0]
    p0.alignment = 2                                 # centred caption, no bullet
    ppr = p0._p.get_or_add_pPr()
    for tag in ("buChar", "buAutoNum", "buNone"):
        for el in ppr.findall(A + tag):
            ppr.remove(el)
    ppr.set("marL", "0"); ppr.set("indent", "0")
    ppr.append(ppr.makeelement(A + "buNone", {}))
    for r in p0.runs:
        r.font.italic = True
        r.font.color.rgb = RGBColor.from_string("4B5563")

    # 6 Results: text left, two figures stacked right
    s = S[5]
    sh = shape(s, "Content Placeholder 2")
    set_bullets(sh, [
        "Same accuracy for all methods (0.22% median error); matches the Cramér–Rao limit",
        "Learned start vs midpoint start: about 2 CPU-s saved per query on the pilot test",
        "Learned start vs Brent: 0.8–1.1 CPU-s saved at the loose tolerance, ~0 at the tight one",
        "Pre-registered test: not positive; 2x target not met (1.2x at matched typical precision)",
        "Follow-up (after the test): MLP guess as a bracket for Brent saves 8–10%",
        "Break-even 2,600–4,500 queries; saving gone for unseen starting states / sensor patterns",
    ], size=14)
    style(sh, 0.6, 1.95, 5.2, 4.9, space_after=8)
    drop(s, "Picture 10")
    w = 7.0
    h1, h2 = fit(FIG / "tolerance_curves.png", w), fit(FIG / "robustness.png", w)
    gap = (4.9 - h1 - h2) / 3
    picture(s, FIG / "tolerance_curves.png", 6.0, 1.95 + gap, width=w)
    picture(s, FIG / "robustness.png", 6.0, 1.95 + 2 * gap + h1, width=w)

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
        ("6", "IEEE paper, slides, live demo video, repository documentation", "Done"),
        ("7", "Larger 200-trajectory test set (optional extra)", "Not done"),
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
        ("9", "Test suite (pytest)", "100", "Y", "29 tests pass"),
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

    check_layout(prs)
    out = HERE / "AFML_project_slides.pptx"
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
