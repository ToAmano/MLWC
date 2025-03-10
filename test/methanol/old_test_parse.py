# coding:utf-8

import unittest
import ml.parse


# https://qiita.com/phorizon20/items/acb929772aaae4f52101

class ParseTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        inputfilename="descripter.inp"
        inputs_list=ml.parse.read_inputfile(inputfilename)
        input_general, input_descripter, input_predict=ml.parse.locate_tag(inputs_list)
        self.var_gen=ml.parse.var_general(input_general)
        self.var_des=ml.parse.var_descripter(input_descripter)
        self.var_pre=ml.parse.var_predict(input_predict)
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_itpfilename(self):
        self.assertEqual("gromacs_input/input_GMX.itp", self.var_gen.itpfilename)

    def test_directory(self):
        self.assertEqual("100ps/", self.var_des.directory)

    def test_filename(self):
        self.assertEqual("gromacs_trajectory_cell.xyz", self.var_des.xyzfilename)

    def test_savedir(self):
        self.assertEqual("100ps/bulk/", self.var_des.savedir)


if __name__ == "__main__":
    unittest.main()
