from importlib.util import find_spec
import unittest


@unittest.skipUnless(find_spec('pypdf'), 'requires PDF runtime')
class CitationCandidateTest(unittest.TestCase):
    def test_change_is_reversible_and_preserves_existing_citation(self):
        from scripts import build_m3w_citation_candidate as m

        source = '\n'.join((m.OLD, r'Another method~\cite{a}.', m.BIB_START, r'\bibitem{a} A.',
                            m.BIB_END))
        output = m.amend(source)
        self.assertEqual(m.used_citations({'manuscript.tex': output.encode()}), ['a', 'wepdtof'])
        self.assertEqual(output.replace(m.NEW, m.OLD).replace(m.BIB_SMALL, m.BIB_START)
                         .replace(m.REFERENCE, '').replace(m.BIB_END+'}', m.BIB_END), source)
        with self.assertRaisesRegex(ValueError, 'original WEPDTOF'):
            m.amend(output)

    def test_rejects_an_uncited_or_missing_reference(self):
        from scripts import build_m3w_citation_candidate as m

        for text in (br'\cite{a}', br'\bibitem{a} Text.'):
            with self.assertRaisesRegex(ValueError, 'do not match'):
                m.used_citations({'manuscript.tex': text})
