import unittest

import numpy as np

class MyTestCase(unittest.TestCase):
    def test_store_load(self):
        s_s = np.random.rand(100, 4)
        u_s = np.random.rand(100, 1)
        np.savez("np_test.npz", s=s_s, u=u_s)

        stuff = np.load("np_test.npz")
        s_l = stuff['s']
        u_l = stuff['u']
        print('\n')
        print(np.max(s_l - s_s))
        print(np.max(u_l - u_s))

if __name__ == '__main__':
    unittest.main()
