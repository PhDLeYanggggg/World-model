from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile, ZipInfo

from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas

from reproducibility.combined_review_supplement.verify import REQUIRED, verify
from scripts.package_m3w_combined_review_supplement import MANTRA, check_pdf, check_replay


class CombinedReviewTests(unittest.TestCase):
    def pdf(self, link=None, author='Anonymous'):
        buf = BytesIO()
        canvas = Canvas(buf, invariant=1)
        canvas.setAuthor(author)
        canvas.drawString(80, 720, 'Anonymous evidence')
        if link:
            canvas.linkURL(link, (80, 718, 120, 733))
        canvas.save()
        return buf.getvalue()

    def test_known_citation_only(self):
        self.assertIn('Anonymous evidence', check_pdf(self.pdf(), 1))
        with self.assertRaisesRegex(ValueError, 'link or action'):
            check_pdf(self.pdf(MANTRA), 1)
        self.assertIn('Anonymous evidence', check_pdf(self.pdf(MANTRA), 1, {MANTRA}))

    def test_pdf_metadata_and_attachments(self):
        with self.assertRaisesRegex(ValueError, 'author metadata'):
            check_pdf(self.pdf(author='Named Author'), 1)
        src = BytesIO(self.pdf())
        writer = PdfWriter(clone_from=src)
        writer.add_attachment('example.txt', b'do not include attachments')
        out = BytesIO()
        writer.write(out)
        with self.assertRaisesRegex(ValueError, 'attached PDF content'):
            check_pdf(out.getvalue(), 1)

    def test_replay_members_reject_paths_and_symlinks(self):
        symlink = ZipInfo('x.py')
        symlink.create_system = 3
        symlink.external_attr = 0o120777 << 16
        for name in ('../other.py', '/other.py', 'x\\y.py', symlink):
            with self.subTest(name=name):
                buf = BytesIO()
                with ZipFile(buf, 'w') as z:
                    z.writestr('verify.py', 'pass\n')
                    z.writestr(name, 'irrelevant')
                with self.assertRaisesRegex(ValueError, 'Unexpected replay archive member'):
                    check_replay(buf.getvalue())

    def test_manifest_detects_extras_and_tampering_before_replay(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {k: k.encode() for k in REQUIRED}
            for name, data in files.items():
                (root/name).write_bytes(data)
            (root/'manifest.json').write_text(json.dumps({k: sha256(v).hexdigest() for k, v in files.items()}))
            extra = root/'unexpected.txt'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'roster'):
                verify(root)
            extra.unlink()
            (root/'technical_appendix.pdf').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(root)
