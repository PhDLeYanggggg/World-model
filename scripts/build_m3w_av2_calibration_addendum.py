"""Append one evidence-bound AV2 diagnostic page to a separate supplement copy."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from build_m3w_budget_addendum import table
from build_m3w_supplement_guide import footer, header, paragraph


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/"outputs/publication_readiness_2026_09"
ADMIN = ROOT/"data/stage_cvpr2027_experiments/submission_admin"
SOURCE = ADMIN/"supplement_with_budget_addendum.pdf"
SOURCE_SHA = "24feba604f08cd4925cf9828f01035b4f69e9520168af7c4e27f23971ae5a5ea"
RECEIPTS = (
    ("av2_incumbent_timing_v1/summary.json", "f81e2af25132c8a5c36fbd5cc95ab206fe8aecd78eed9abf7a7a0bbdd7a80b71"),
    ("av2_incumbent_timing_v1/capacity_attribution.json", "b3e21943123edfda43071990b8054d40d5b36872412775becc0f82ca30bcfcab"),
    ("av2_replay_hurdle_v1/summary.json", "7d3601093eb7b2554b0d33251a044acb0be065b8abfcfc25c36d3febbfa52686"),
)
OUTLINE = "AV2 timing and calibration"


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def load():
    rows = []
    for name, expected in RECEIPTS:
        path = REPORTS/name
        if digest(path) != expected:
            raise ValueError(f"Unexpected evidence version: {name}")
        r = json.loads(path.read_bytes())
        if not r["original_2pct"] or r["reserved_outcomes_opened"] or r["independent_confirmation"]:
            raise ValueError("Expected development-only results under the original rule")
        rows.append(r)
    timing, capacity, gate = rows
    if (timing["counts"]["two_slots"] != gate["counts"]["two_slots"]
            or timing["counts"]["replay_top"] != gate["counts"]["matched"]
            or timing["counts"]["local_available"] != capacity["counts"]["local_timing"]
            or not gate["identical_replay_paths"]):
        raise ValueError("Fixed paths and counts did not replay")
    return timing, capacity, gate


def page(evidence, number):
    _, capacity, gate = evidence
    z, g = capacity["counts"], gate["counts"]
    result = BytesIO()
    c = canvas.Canvas(result, pagesize=letter, invariant=1)
    c.setTitle("AV2 timing and calibration")
    c.setAuthor("")
    header(c, "ARGOVERSE 2 DEVELOPMENT DIAGNOSTIC", "What timing can change")
    y = paragraph(c, f"We kept the original two edits and their peak displacements fixed. "
                  f"The {g['two_slots']:,} views come from 340 development scenarios in six cities, "
                  "with eight observed steps, twelve forecast steps and three training seeds. "
                  "The same 2% harm checks and both motion floors apply to each path.", 645)
    y = table(c, [["Alternative to the same independent path", "Views"],
        ["Only change timing", str(z["local_timing"])],
        ["Any forecast source; same actors and assigned peaks", str(z["same_actor_and_peak"])],
        ["Any source; either peak assignment to those actors", str(z["same_actors"])],
        ["Any recipients in the full menu", str(z["all_recipients"])]], [373, 115], y)
    y = paragraph(c, f"The first table counts opportunities with hindsight. A past-only replay rule "
                  f"selected {g['proposals']} substitutions, of which {g['anchor_eligible']} had an "
                  f"eligible independent path and {g['matched']} passed both arms' checks. "
                  "The following scores change the release order only, not the chosen paths.", y)
    rules = (("Replay score", "replay"), ("Replay times reference probability", "anchor_replay"),
             ("Direct matched probability", "direct"), ("Factored matched probability", "factored"))
    rows = [["Score on the fixed replay proposals", "Calibration", "Held pass / release"]]
    for name, k in rules:
        n = g[f"{k}_released"]
        rows.append([name, str(gate["calibration"][k]), f"{g[f'{k}_pass']}/{n}" if n else "No release"])
    y = table(c, rows, [258, 92, 138], y)
    case, = gate["releases"]["factored"]
    y = paragraph(c, f"The single factored release improved on its independent path by "
                  f"{case['relative'][0]:.2f} percentage points of conservative trajectory gain. "
                  f"It nevertheless lost to OLS4 ({case['gain'][1]:.2f} points) and failed the "
                  "easy-case risk check. No released replay substitution passed. The result "
                  "distinguishes candidate supply from reliable release on new development cities.", y)
    if y < 67:
        raise ValueError("AV2 addendum overflowed")
    footer(c, number)
    c.save()
    result.seek(0)
    return result


def verify(original, final):
    n = len(original.pages)
    if len(final.pages) != n+1:
        raise ValueError("Expected exactly one new page")
    def annotations(p):
        return p["/Annots"].get_object() if "/Annots" in p else []

    for x, y in zip(original.pages, final.pages[:n], strict=True):
        if (x.mediabox != y.mediabox or x.extract_text() != y.extract_text()
                or x.get_contents().get_data() != y.get_contents().get_data()
                or len(annotations(x)) != len(annotations(y))):
            raise ValueError("A previous supplement page changed")
    def dest(r):
        return {k: r.get_destination_page_number(v) for k, v in r.named_destinations.items()}
    if dest(original) != dest(final):
        raise ValueError("Named destinations changed")
    for a, b in zip(annotations(original.pages[0]), annotations(final.pages[0]), strict=True):
        x, y = a.get_object()["/Dest"], b.get_object()["/Dest"]
        j = next(i for i, p in enumerate(original.pages) if p.indirect_reference == x[0])
        if final.pages[j].indirect_reference != y[0] or x[1:] != y[1:]:
            raise ValueError("Evidence-guide link changed")
    outline = {q.title: final.get_destination_page_number(q) for q in final.outline if not isinstance(q, list)}
    p = final.pages[-1]
    if outline.get(OUTLINE) != n or annotations(p) or final.metadata.author:
        raise ValueError("New page must be self-contained and anonymous")
    text = p.extract_text()
    if any(k not in text for k in ("What timing can change", "51", "56", "61", "0/1", "1,020", "OLS4")):
        raise ValueError("New PDF does not contain the intended results")
    return dict(pages=len(final.pages), original_pages=n, preserved_named_destinations=len(dest(original)))


def build(output):
    if output.exists() or output.resolve() == SOURCE.resolve():
        raise FileExistsError("Refusing to overwrite a submission artifact")
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError("The supplement differs from the verified source")
    original = PdfReader(SOURCE)
    writer = PdfWriter(clone_from=original)
    writer.add_page(PdfReader(page(load(), len(original.pages)+1)).pages[0])
    writer.add_outline_item(OUTLINE, len(original.pages))
    writer.add_metadata({"/Title": "Supplementary material", "/Author": "", "/Subject": ""})
    pending = output.with_suffix(".pending")
    writer.write(pending)
    receipt = verify(original, PdfReader(pending))
    pending.replace(output)
    print(json.dumps(dict(**receipt, sha256=digest(output)), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    build(p.parse_args().output)
