import unittest

import ltn.logictensornetworks as ltn


class MyTestCase(unittest.TestCase):
    def test_something(self):
        c = ltn.constant('a', min_value=[0.] * 20)
        assert hasattr(c, '_doms')
        assert c._doms == []
        assert ltn.CONSTANTS['a'] is c


if __name__ == '__main__':
    unittest.main()
