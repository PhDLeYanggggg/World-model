import unittest

from scripts.build_m3w_metric_scope_candidate import SCOPE, revise


class MetricScopeTests(unittest.TestCase):
    def test_only_the_exact_two_paragraphs_change(self):
        a, b = SCOPE
        protected = '\\begin{abstract}Text\\end{abstract}\n\\section*{LLM Usage Statement}\nUnchanged'
        text = a+'\n'+b+'\n'+protected
        self.assertEqual(revise(text), SCOPE[a]+'\n'+SCOPE[b]+'\n'+protected)

    def test_unverified_or_duplicate_paragraph_fails(self):
        a, b = SCOPE
        for text in (a, a+'\n'+b+'\n'+b, a+'\n'+SCOPE[b]):
            with self.subTest(case=text[:25]), self.assertRaises(ValueError):
                revise(text)
