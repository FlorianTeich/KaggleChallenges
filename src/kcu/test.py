"""
Tests
"""
import os
import unittest

from kcu import visualization


class TestVisualization(unittest.TestCase):
    """
    Test Visualizations
    """

    def test_show_image_from_path(self):
        """
        Test for showing an image from a path
        """
        self.assertEqual(
            visualization.show_image_from_path(
                os.path.dirname(__file__) + "/../../data/red_panda.jpeg", hidden=True
            ),
            1,
        )
