from unittest import TestCase

from neat.config import Config


class TestConfig(TestCase):

    def setUp(self):
        self.file_name = 'conf_tester.conf'

    def test_config_initializes_with_test_file(self):
        cfg = Config(self.file_name)

    def test_config_initializes_without_test_file(self):
        cfg = Config()