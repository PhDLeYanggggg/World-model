# Word draft and CVPR review PDF

The single-column Word draft is for author editing. It is exported
from the verified v218 LaTeX source, without the review ruler. It
retains three native Word tables, three native display equations,
three figures from the original plots, and 21 references. The
disclosure and results are unchanged. The 11-page Word render is
not the submission layout.

The [CVPR 2027 author guidelines][guidelines] require the official
layout and at most eight pages of content, including figures and
tables. References and the LLM Usage Statement may extend past that
limit. The [2027 author kit][kit], in `sec/1_intro.tex`, specifies
the margin ruler for review and removes it for the camera-ready copy.
Its `cvpr.sty` SHA-256 is
`1cd7f434b13d41c475df4a1901a5d82d9ca8b80b8cf6205dc4602a5c164286b4`,
the exact style used for the existing v218 build.

The unchanged v218 main PDF is anonymous, US Letter, double column,
and eight pages total. Its conclusion, LLM Usage Statement and start
of the references are on page 7; page 8 continues the references. The
official review markings are present, and no external URI annotations
were found. This is a file-format check, not an OpenReview submission
or an audit of the author's profile.

The nine-page supplement retains black serif headings and black and
white tables. Only the plotted data on the two figure pages use their
original color palette. The curves, scales and extracted text are
unchanged. The anonymous numerical replay was checked again.

The local files are in
`data/stage_cvpr2027_experiments/submission_admin/local_handoff_v218_color_figures/`:
`paper_editable.docx`, `paper.pdf`, `supplement.pdf`, and `replay.zip`.
No document was uploaded. The Word export uses Pandoc, Poppler and
`python-docx`; the supplement builder uses `pypdf` and ReportLab.

~~~sh
python scripts/build_m3w_serif_supplement.py --color-figures
python scripts/export_m3w_v218_word.py
~~~

[Supplement and replay receipt](receipt.json) |
[Official deadlines][dates].

[guidelines]: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
[kit]: https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip
[dates]: https://cvpr.thecvf.com/Conferences/2027/Dates
