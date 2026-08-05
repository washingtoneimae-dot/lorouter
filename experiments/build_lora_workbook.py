"""
build_lora_workbook.py

Builds lorouter_results.xlsx -- the canonical results workbook for the
lorouter branch. Numbers are this branch's canonical run (the actual
outputs of the experiments listed under 'source' below, run 2026-08-05 on
the lorouter branch). The scripts remain the source of truth; this
workbook is the summary artifact, consistent with how performance_data.xlsx
is produced on the v2 branch.

Sheets:
  Summary          -- every experiment, one line, key numbers + source
  StandinBenchmark -- 4-way strategy comparison (5 seeds, brick 2)
  RealLoRA         -- loss matrix + routing (SmolLM2-135M, 10 epochs)
  UnseenTask       -- arm-by-arm oracle agreement + loss gap (answer-NLL)
  EightAdapter     -- 8-adapter pool: accuracy, separation, isolation
  Corpus           -- brick growth + calibration thresholds (brick 2 vs 3)

Charts: one per sheet (native openpyxl charts, ink/amber palette).

Run: python3 experiments/build_lora_workbook.py
Requires: openpyxl
"""
import openpyxl
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from pathlib import Path

OUT = Path(__file__).resolve().parent / "lorouter_results.xlsx"

INK = "0B0C0E"
CREAM = "F5EFE6"
AMBER = "FFB454"
TEAL = "2DD4BF"
RED = "F87171"

wb = openpyxl.Workbook()

HDR_FILL = PatternFill("solid", fgColor=AMBER)
HDR_FONT = Font(color=INK, bold=True, size=11)
TITLE_FONT = Font(color=CREAM, bold=True, size=14)
SUB_FONT = Font(color="A89F8D", italic=True, size=9)


def sheet(name, title, subtitle, headers, rows):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    ws["A2"] = subtitle
    ws["A2"].font = SUB_FONT
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=4, column=c, value=h)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
    for r, row in enumerate(rows, 5):
        for c, v in enumerate(row, 1):
            ws.cell(row=r, column=c, value=v)
    ws.freeze_panes = "A5"
    return ws


def bar(ws, title, cat_ref, data_refs, labels, colors, anchor="H4"):
    ch = BarChart()
    ch.type = "col"
    ch.title = title
    ch.style = 10
    for ref, lab, col in zip(data_refs, labels, colors):
        ch.add_data(ref, titles_from_data=False)
        s = ch.series[-1]
        s.tx = openpyxl.chart.series.SeriesLabel(v=lab)
        s.graphicalProperties.solidFill = col
    ch.set_categories(cat_ref)
    ch.height = 9
    ch.width = 20
    ch.gapWidth = 80
    ws.add_chart(ch, anchor)
    return ch


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
summary_rows = [
    ["Stand-in benchmark (brick 2, 5 seeds, 56 test/seed)", "profile 96.4% | centroid 97.9% | learned 96.4% | random 19.3%", "experiments/benchmark.py"],
    ["Swap isolation (text setting, stand-in)", "0 flips on other domains (weak-replacement swap)", "experiments/benchmark.py"],
    ["Real-LoRA integration (SmolLM2-135M, rank 8, 10 ep)", "routing 96.4% (54/56); differentiation 2/4 diagonal; finance/law adapters favor code (base priors)", "experiments/real_lora_integration.py"],
    ["Unseen-task (education held out, 14 queries)", "routing stable: education->finance 10/14 (real + stand-in arms agree); oracle agreement 42.9%; loss gap == random (quality OPEN)", "experiments/unseen_task_generalization.py"],
    ["LORAUTER-style exemplar signal (answer-NLL oracle)", "V1 42.9% / +0.13% | V2a 28.6% / +0.12% (all->code, lexical artifact) | V2b 35.7% / +0.12% | V3 aligned 14/14 routed, +0.04% | ceiling 100%", "experiments/lora_exemplar_routing.py"],
    ["8-adapter pool (2/domain, disjoint data)", "domain accuracy 96.4% (unchanged); profile min cosine 0.996 (near-duplicate adapters); swap flips 0/56", "experiments/eight_adapter_space.py"],
    ["Corpus brick 3 (664 ex, 25% calib split)", "p99 clean-only threshold 0.30->0.84, false-capture 4.76%->0.00% vs brick 2 (degeneracy resolved)", "scripts/build_moat_corpus.py + scripts/moat_calibration_trial.py"],
]
ws = sheet("Summary", "LOROUTER -- canonical results", "Canonical run 2026-08-05, lorouter branch. Scripts are the source of truth.",
           ["Experiment", "Key result", "Source script"], summary_rows)
ws.column_dimensions["A"].width = 52
ws.column_dimensions["B"].width = 95
ws.column_dimensions["C"].width = 42

# ----------------------------------------------------------------------
# Stand-in benchmark
# ----------------------------------------------------------------------
ws = sheet("StandinBenchmark", "Stand-in benchmark -- adapter selection accuracy",
           "Brick 2 corpus machinery, 5 seeds, 56 test inputs/seed. Random floor 19.3%.",
           ["strategy", "acc seed1", "seed2", "seed3", "seed4", "seed5", "mean"],
           [
               ["profile (lorouter)", 0.964, 0.964, 0.964, 0.964, 0.964, 0.964],
               ["centroid (LORAUTER-style)", 0.982, 0.982, 0.982, 0.982, 0.964, 0.979],
               ["learned router (MoLE line)", 0.964, 0.964, 0.964, 0.964, 0.964, 0.964],
               ["random", 0.161, 0.214, 0.161, 0.214, 0.214, 0.193],
           ])
bar(ws, "Adapter selection accuracy by strategy",
    Reference(ws, min_col=1, min_row=5, max_row=8),
    [Reference(ws, min_col=7, min_row=5, max_row=8)],
    ["mean accuracy"], [AMBER])
ws["G5"].number_format = "0.0%"
ws["G6"].number_format = "0.0%"
ws["G7"].number_format = "0.0%"
ws["G8"].number_format = "0.0%"

# ----------------------------------------------------------------------
# Real LoRA
# ----------------------------------------------------------------------
ws = sheet("RealLoRA", "Real-LoRA integration -- calibration loss matrix",
           "SmolLM2-135M-Instruct, rank 8, 10 epochs, QA from brick 3. Loss = answer-conditional NLL on calibration (question masked). Lower = better.",
           ["adapter \\ calib domain", "code", "education", "finance", "law"],
           [
               ["code", 4.047, 5.259, 6.046, 6.090],
               ["education", 3.862, 3.749, 4.999, 4.835],
               ["finance", 3.892, 4.327, 4.670, 5.163],
               ["law", 4.308, 4.973, 5.850, 5.020],
           ])
ws["A9"] = "routing accuracy (profile router): 96.4% (54/56) | diagonal dominance: 2/4 | source: experiments/real_lora_integration.py"
ws["A9"].font = SUB_FONT
bar(ws, "Calibration loss by adapter and domain",
    Reference(ws, min_col=1, min_row=5, max_row=8),
    [Reference(ws, min_col=c, min_row=5, max_row=8) for c in (2, 3, 4, 5)],
    ["code", "education", "finance", "law"],
    [TEAL, AMBER, RED, "A89F8D"])
for r in range(5, 9):
    for c in range(2, 6):
        ws.cell(row=r, column=c).number_format = "0.000"

# ----------------------------------------------------------------------
# Unseen task
# ----------------------------------------------------------------------
ws = sheet("UnseenTask", "Unseen-task generalization -- arms compared",
           "Education fully held out, 14 test queries. Oracle = per-query lowest-answer-NLL adapter. Loss gap vs oracle; random expectation +0.13%.",
           ["arm", "oracle agreement", "loss gap vs oracle", "routing distribution"],
           [
               ["V1 per-query profile routing", 0.429, 0.0013, "finance 10, code 2, law 2"],
               ["V2a task-level, exemplars=5", 0.286, 0.0012, "code 14 (lexical artifact)"],
               ["V2a task-level, exemplars=10", 0.286, 0.0012, "code 14"],
               ["V2b per-query embeddings, ex=5", 0.357, 0.0012, "finance 6, code 4, law 4"],
               ["V2b per-query embeddings, ex=10", 0.357, 0.0012, "finance 6, code 4, law 4"],
               ["V3 aligned-adapter control", 0.000, 0.0004, "education 14/14 (routing perfect)"],
               ["oracle-best (ceiling)", 1.000, 0.0000, "finance 5, law 5, code 4"],
           ])
for r in range(5, 12):
    ws.cell(row=r, column=2).number_format = "0.0%"
    ws.cell(row=r, column=3).number_format = "0.00%"
bar(ws, "Oracle agreement by routing arm",
    Reference(ws, min_col=1, min_row=5, max_row=11),
    [Reference(ws, min_col=2, min_row=5, max_row=11)],
    ["oracle agreement"], [AMBER])

# ----------------------------------------------------------------------
# Eight adapter
# ----------------------------------------------------------------------
ws = sheet("EightAdapter", "Eight-adapter pool (2/domain)",
           "Disjoint training data per variant, different answer wording. Domain-level accuracy is the meaningful metric (both variants correct).",
           ["metric", "value", "note"],
           [
               ["domain-level routing accuracy", 0.964, "54/56; identical to 4-adapter pool"],
               ["profile min pairwise cosine", 0.996, "near-duplicate adapters -> near-identical profiles"],
               ["profile mean pairwise cosine", 0.998, "separation bounded by adapter diversity"],
               ["swap isolation (code_A swapped)", 0, "0/56 routing flips on other domains"],
               ["variant split (routed, info only)", "code 0A/13B, edu 16A/0B, fin 0A/14B, law 13A/0B", "router consistently prefers one variant per domain"],
           ])
ws["A10"] = "source: experiments/eight_adapter_space.py"
ws["A10"].font = SUB_FONT
bar(ws, "Routing accuracy: 4-adapter vs 8-adapter pool",
    Reference(ws, min_col=1, min_row=5, max_row=6),
    [Reference(ws, min_col=2, min_row=5, max_row=6)],
    ["accuracy"], [TEAL])
ws["B5"].number_format = "0.0%"

# ----------------------------------------------------------------------
# Corpus growth
# ----------------------------------------------------------------------
ws = sheet("Corpus", "Moat corpus growth + calibration stability",
           "Bricks built by scripts/build_moat_corpus.py (seeded, deterministic). Thresholds from scripts/moat_calibration_trial.py (gate: education).",
           ["brick", "total examples", "boundaries", "calibration split", "p99 clean-only thr", "p99 false-capture", "p95 clean-only thr", "p95 clean+boundary thr"],
           [
               ["brick 1", 181, 17, "70/15/15 (n~32)", None, None, None, None],
               ["brick 2", 408, 48, "70/15/15 (n=80)", 0.3048, 0.0476, 0.1361, 0.9353],
               ["brick 3", 664, 64, "60/25/15 (n=180)", 0.8439, 0.0000, 0.1283, 0.6029],
           ])
ws["A9"] = "brick 2 -> 3: small-n p99 degeneracy resolved (threshold 0.30->0.84, false-capture 4.76%->0.00%); boundary value now visible at p95."
ws["A9"].font = SUB_FONT
lc = LineChart()
lc.title = "Corpus growth across bricks"
lc.style = 12
lc.height = 9
lc.width = 20
data = Reference(ws, min_col=2, min_row=4, max_row=7)
cats = Reference(ws, min_col=1, min_row=5, max_row=7)
lc.add_data(data, titles_from_data=True)
lc.set_categories(cats)
ws.add_chart(lc, "H4")
bar(ws, "Calibration thresholds: brick 2 vs brick 3 (p95)",
    Reference(ws, min_col=1, min_row=5, max_row=6),
    [Reference(ws, min_col=7, min_row=5, max_row=6),
     Reference(ws, min_col=8, min_row=5, max_row=6)],
    ["clean-only", "clean+boundary"], [RED, TEAL], anchor="H23")

# drop the default sheet
if "Sheet" in wb.sheetnames:
    del wb["Sheet"]

wb.save(OUT)
print(f"wrote {OUT}")
print("sheets:", wb.sheetnames)
