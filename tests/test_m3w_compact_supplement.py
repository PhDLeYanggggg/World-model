from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(find_spec('reportlab') and find_spec('pypdf'), 'requires PDF runtime')
class CompactSupplementTest(unittest.TestCase):
    def test_renders_internal_bookmarks_and_rejects_external_link(self):
        from scripts import build_m3w_compact_supplement as m

        with TemporaryDirectory() as root:
            for external in (False, True):
                path = Path(root)/f'{external}.pdf'
                pdf = m.canvas.Canvas(str(path), invariant=1)
                pdf.setAuthor('')
                for i, heading in enumerate(m.HEADINGS, 1):
                    p = m.Page(pdf, i, heading)
                    p.text('Anonymous evidence')
                    if i == 2 and external:
                        pdf.linkURL('https://example.com', (20, 20, 50, 50))
                    p.close()
                pdf.save()
                if external:
                    with self.assertRaisesRegex(ValueError, 'annotation or link'):
                        m.verify(path)
                else:
                    m.verify(path)

    def test_format_keeps_signed_comparison_and_frozen_interval(self):
        from scripts.build_m3w_compact_supplement import interval

        self.assertEqual(interval(dict(estimate=-.04501, ci95=[-.11323, .01214]), 5),
                         '-0.04501 [-0.11323, +0.01214]')
