# Supplementary PDF: monochrome serif edition

The nine-page supplementary PDF uses black serif text, numbered
sections, light horizontal table rules and centered page numbers.
The original charts retain their geometry and data, with grayscale
marks and new page headings. The corrected v218 methods, results
and WEPDTOF terminology are retained.

The local matched files are in
`data/stage_cvpr2027_experiments/submission_admin/local_handoff_v218_serif/`.
The main `paper.pdf` is byte-for-byte identical to v218, including
its LLM Usage Statement. The updated `replay.zip` contains the
monochrome PDF and reproduces the same numeric readout.

```sh
python scripts/build_m3w_serif_supplement.py
```

The build uses pypdf and ReportLab plus the verified local evidence.
[Build and replay receipt](receipt.json).
