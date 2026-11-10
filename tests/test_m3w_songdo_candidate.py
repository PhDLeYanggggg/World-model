import copy
import unittest

from scripts import build_m3w_songdo_candidate as paper


class SongdoPaperTests(unittest.TestCase):
    def test_matched_and_released_counts_are_rendered_separately(self):
        score = paper.verified_evidence()
        note = paper.paragraph(score)
        self.assertIn('40 flights', note)
        self.assertIn('160 flights', note)
        self.assertIn('Joint released 0/480 views; independent passed risk on 5/15', note)
        self.assertIn('[-0.001499,+0.000237]', note)

    def test_wrong_cohort_or_released_joint_does_not_build(self):
        score = copy.deepcopy(paper.verified_evidence())
        score['released']['joint']['risk']['edited'] = 1
        with self.assertRaisesRegex(ValueError, 'complete frozen Songdo finding'):
            paper.paragraph(score)
