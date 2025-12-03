import unittest
import numpy as np
import os
import torch

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

    def test_rand(self):
        low_vec = -np.array([0.1,2.0,0.5,12.0])
        high_vec = np.array([0.1, 2.0, 0.5, 12.0])
        one_pull = np.random.uniform(low_vec, high_vec, size=(10,4))
        print("\n")
        print(one_pull)

    def test_files(self):
        for f in os.listdir("trajectories_test"):
            print(f)

    def test_numpy_append(self):
        all_arr = []
        for i in range(0,3):
            arr = np.random.rand(8, 4)
            all_arr.append(arr)
        np_all_arr = np.concatenate(all_arr,0)
        print(np_all_arr)

    def test_numpy_bcast(self):
        norm = [1, 2, 3, 4]
        t = np.array([[1,1,1,1],[2,2,2,2]])
        print (t/norm)

    def test_random(self):
        uvec = np.array([0.0, np.pi / 4, 0.0, 0.0])
        s0_batch = np.squeeze(np.random.uniform(-uvec, uvec, size=(128, 4)))
        print(s0_batch.size)

    def test_np_and(self):
        #umm, per-element logical ops
        a = np.array([True, False, True])
        b = np.array([False, True, True])
        c = np.array([True, False, True])
        print(a & b & c)

    def test_tensor_math(self):
        t = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        norm = [1.0, 2.0, 3.0, 4.0]
        tt = torch.tensor(t)
        print(t/norm)
        print(tt/torch.tensor(norm))

    def test_np_maximum(self):
        t1 = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
        t2 = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
        t1d = t1+np.array([1.0, 1.0, 1.0, 1.0])
        tm = np.maximum(t1d, t2)
        print(tm)

    def test_np_argmax(self):
        # what type comes back from this thing?
        t = np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 3.0, 4.0]])
        tam = np.argmax(t)
        print(tam)

if __name__ == '__main__':
    unittest.main()
