import pytest

from scripts.build_m3w_native_paper_v10 import latex_document


def test_ast_preserves_aligned_math_and_escapes_table_cells():
    text = """# Forecast Study

## Abstract

Only measured results.

## 1. Method

$$
\\begin{aligned}
B&=1,\\\\
H&=2.
\\end{aligned}
$$

### Table 1. Observed development data

| Input | Gain |
|---|---:|
| No_image | 2% |

## References

1. https://example.org/paper
"""
    tex = latex_document(text)
    assert r'\begin{aligned}' in tex
    assert r'\end{aligned}' in tex
    assert r'No\_image & 2\%' in tex
    assert r'\caption{Observed development data}' in tex
    assert r'\section{Method}' in tex
    assert r'\clearpage\section*{References}' in tex
    assert tex.count(r'\begin{abstract}') == tex.count(r'\end{abstract}') == 1


def test_paper_figures_reject_external_or_data_images():
    with pytest.raises(ValueError, match='generated scalar figures'):
        latex_document('# Study\n\n![Private image](data/image.png)\n')


def test_tables_require_explicit_captions():
    with pytest.raises(ValueError, match='caption'):
        latex_document('# Study\n\n| Input | Gain |\n|---|---|\n| A | B |\n')
