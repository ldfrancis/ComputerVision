
from gans.discriminator import Discriminator
import unittest

class TestDesciminator(unittest.TestCase):

    def test_initialization(self):
        disc = Discriminator()
        disc.save_model()

    


if __name__ == '__main__':
    unittest.main()
