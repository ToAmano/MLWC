from mlwc.cpmd.assign_wcs.assign_wcs_torch import atoms_wan
from mlwc.ml.dataset.mldataset_abstract import AbstractDataset


class GenericMolecularDataset(AbstractDataset):

    def __init__(self, input_atoms_wan_list: list[atoms_wan], transforms=None):
        # TODO :: ファイルからatoms_wanデータの読み込みを実現する
        self.data = input_atoms_wan_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO:: ファイルから原子番号、座標、結合、物性値などを読み込む
        # モデルのことは考えず、純粋なデータオブジェクトを作成
        molecular_data = self.data[idx]

        # もしTransformが指定されていれば、ここで適用する
        if self.transforms:
            molecular_data = self.transforms(molecular_data)

        return molecular_data
