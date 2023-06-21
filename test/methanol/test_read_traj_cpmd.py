# coding:utf-8

import unittest
import ml.parse
import numpy as np
import cpmd
import cpmd.read_traj_cpmd


# https://qiita.com/phorizon20/items/acb929772aaae4f52101

class ReadTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        self.filepointer=open("gromacs_30.xyz")
        self.NUM_ATOM=384
        pass

    def tearDown(self):
        # 終了処理
        self.filepointer.close()
        pass

    def test_read_traj_cpmd_1_firstconf(self):
        '''
        fileを1行目から読み込む．
        '''
        # 1つ目のconfig
        symbols, positions, self.filepointer = cpmd.read_traj_cpmd.raw_cpmd_read_xyz(self.filepointer, self.NUM_ATOM)
        self.assertEqual(symbols[1], "O")
        self.assertEqual(positions[1][0],3.64500570)
        self.assertEqual(symbols[383], "H")
        self.assertEqual(positions[383][0],8.53548241)

        # 2つ目のconfig
        symbols, positions, self.filepointer = cpmd.read_traj_cpmd.raw_cpmd_read_xyz(self.filepointer, self.NUM_ATOM)
        self.assertEqual(symbols[1], "O")
        self.assertEqual(positions[1][0],3.64225674)
        self.assertEqual(symbols[383], "H")
        self.assertEqual(positions[383][0],8.53945923)



if __name__ == "__main__":
    unittest.main()
