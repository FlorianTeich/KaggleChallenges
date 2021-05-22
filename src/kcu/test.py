import unittest
from kcu import visualization
import os


class TestVisualization(unittest.TestCase):
    def test_show_image_from_path(self):
        self.assertEqual(
            visualization.show_image_from_path(
                os.path.dirname(__file__) + "/../../data/red_panda.jpeg", hidden=True
            ),
            1,
        )
