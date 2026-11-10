import unittest

from scripts import build_m3w_related_work_candidate as paper
from scripts.build_m3w_citation_candidate import used_citations


class RelatedWorkTests(unittest.TestCase):
    def test_revised_prior_work_and_references_leave_claim_and_disclosure_intact(self):
        lead = '\\begin{abstract}Frozen numbers\\end{abstract}\n'
        declaration = '\\section*{LLM Usage Statement}\nUnchanged.\n'
        sources = ('schoeller', 'mao', 'yu', 'natra', 'realtraj', 'martins')
        refs = '\n'.join(r'\bibitem{'+k+'} Reference' for k in sources)+'\n'
        old = lead+paper.OLD+'\n'+declaration+refs+paper.BIB_END
        new = paper.revise(old)
        self.assertEqual(new, lead+paper.NEW+'\n'+declaration+refs+paper.ADDED+paper.BIB_END)
        self.assertEqual(used_citations({'main.tex': new.encode()}), sorted([*sources, 'xuloss', 'wangpair']))

    def test_revisions_fail_closed_on_other_versions(self):
        for text in (paper.NEW+paper.BIB_END, paper.OLD+paper.BIB_END+paper.ADDED):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                paper.revise(text)
