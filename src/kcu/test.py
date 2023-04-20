"""
Tests
"""
import os
import unittest

from initialization import setup_db
from kcu import visualization


class TestSetups(unittest.TestCase):
    """
    Test Setups
    """

    def test_setup_titanic(self):
        """
        Test for database setup
        """
        res = setup_db.setup_db_titanic()
        self.assertEqual(1, res)
