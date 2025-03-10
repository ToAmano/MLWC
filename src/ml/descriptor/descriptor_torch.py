import ase
import numpy as np
import torch  
from typing import Literal # for annotation
import numpy.typing as npt # for annotation
# from torchtyping import TensorType # for annotation
from jaxtyping import Float
from ml.descriptor.descriptor_abstract import Descriptor_abstract
from cpmd.pbc.pbc import pbc
from cpmd.pbc.pbc_torch import pbc_3d_torch

from cpmd.descripter import cutoff_func_torch

class Descriptor_torch_bondcenter(Descriptor_abstract):
    """method for calculating descriptor for both bondcenter and lonepair

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        super().__init__()
    
    @classmethod
    def calc_sorted_generalized_coordinate(cls,
                                        distance_array_3d: Float[torch.Tensor, "bondcent atom distance"], # distnace:3
                                        Rcs:float,Rc:float
                                        )->Float[torch.Tensor, "bondcent atom distance+1"]: # distnace:4
        """sort distance and sij

        Args:
            d (np.ndarray): distance
            s (np.ndarray): sij

        Returns:
            Tuple[np.ndarray,np.ndarray]: sorted distance and sij
        """
                #for C atoms (all) 
        d_r = torch.sqrt(torch.sum(distance_array_3d**2,axis=2)) # 距離の二乗から1乗(r)を導出
        s_r = cutoff_func_torch(d_r,Rcs,Rc)
        # 
        tmp = s_r[:,:,None]*distance_array_3d/d_r[:,:,None] # (s(r)*x/r, s(r)*y/r, s(r)*z/r)
        dij  = torch.cat([s_r[:,:,None],tmp],dim=2) # s(r), s(r)*x/r, s(r)*y/r, s(r)*z/r
        # 1/rの大きい順にnum_atom軸に沿ってソート
        sorted_indices = torch.argsort(s_r, dim=1, descending=True) # s_r はdij[..., 0]でもok
        # ソートされたテンソルを取得する
        dij = torch.gather(dij, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 4))
        return dij
    
    @classmethod
    def fix_length_desc(cls,desc:torch.Tensor,MaxAt:int,device)->torch.Tensor:
        """fix length of descriptor

        Args:
            desc (torch.tensor): descriptor
            MaxAt (int): max length of descriptor

        Returns:
            torch.tensor: fixed length descriptor
        """
        # 要素数が4*MaxAtよりも小さい場合、4*MaxAtになるように0埋めする
        if desc.size(1) < 4*MaxAt:
            padding = torch.zeros(desc.size(0), 4*MaxAt - desc.size(1)).to(device)
            desc = torch.cat([desc, padding], dim=1)
        return desc[:,:MaxAt*4] 
    
    @classmethod
    def calc_descriptor(cls,
                        atoms:ase.Atoms,
                        bond_centers:npt.NDArray[np.float64], # [bondcent,3]
                        list_atomic_number:list[int] = [6,1,8], # [C,H,O]
                        list_maxat:list[int] = [24,24,24], #  [C,H,O] 
                        Rcs:float=4.0, # in Ang
                        Rc:float=6.0,  # in Ang
                        device:str="cpu" # "cuda", "cpu" or "mps"
                        )->np.ndarray:
        if len(bond_centers.shape) != 2 or bond_centers.shape[1] != 3:
            raise ValueError(f"bond_centers should be 2D array. bond_centers.shape should be (bondcent,3) :: {bond_centers.shape}")
        list_mol_coords  = np.array(atoms.get_positions(),dtype="float32")
        list_atomic_nums = np.array(atoms.get_atomic_numbers(), dtype="int32") # 先にint32に変換しておく必要あり
        # convert numpy array to torch
        list_mol_coords  = torch.tensor(list_mol_coords).to(device)
        list_atomic_nums = torch.tensor(list_atomic_nums).to(device)
        bond_centers     = torch.tensor(np.array(bond_centers,dtype="float32")).to(device)
        # 分子座標-ボンドセンター座標を行列の形で実行する
        # list_mol_coords:: [Frame,]
        # mat_ij = atom_i - atom_
        mat_atom = list_mol_coords[None,:,:].repeat(len(bond_centers),1,1) # 原子座標
        mat_bc   = bond_centers[None,:,:].repeat(len(list_mol_coords),1,1) # ボンドセンター座標
        mat_bc   = torch.transpose(mat_bc, 1,0)
        drs: Float[torch.Tensor, "bondcent atom distance=3"] = (mat_atom - mat_bc) # drs:: [bondcent,Atom,3]

        # apply pbc to drs
        dist_wVec:torch.Tensor = pbc(pbc_3d_torch).compute_pbc(vectors_array = drs,
                                                    cell = np.array(atoms.get_cell()),
                                                    device = device)# [bondcent,Atom,3]

        # get atomic numbers from atoms
        # ! CAUTION:: index is different from raw_get_desc_bondcent_allinone
        list_descs = []
        for at,MaxAt in zip(list_atomic_number,list_maxat): # at=6,1,8
            atoms_indx = torch.argwhere(list_atomic_nums==at).reshape(-1) # get index for each atom
            #for C atoms (all) 
            #C原子のローンペアはありえないので原子間距離ゼロの判定は省く
            dist_atoms:torch.Tensor = dist_wVec[:,atoms_indx,:]
            # 距離0の原子を省く．これを入れておけば，lone pairにも対応できる．
            dist_atoms:torch.Tensor = dist_atoms[torch.sum(dist_atoms**2,axis=2)>0.0001].reshape((len(bond_centers),-1,3)) #各行に１つづつ重複した原子が存在するはず
            # calculate generalized coordinate s(r)*(1,x/r,y/r,z/r)
            dij:torch.Tensor = cls.calc_sorted_generalized_coordinate(dist_atoms,Rcs,Rc) # [bondcent,Atom,4]
            # 4d vectorのatomと最後の軸を潰して2次元化する．
            #if len(neighbor list) < MaxAt, zero-padding to MaxAt. 
            dij_descs = cls.fix_length_desc(dij.reshape((len(bond_centers),-1)),MaxAt,device)
            list_descs.append(dij_descs)
        return np.concatenate(list_descs, axis=1)

    @classmethod
    def forward(cls,
                        atomic_coordinate:np.ndarray,
                        atomic_numbers:np.ndarray,
                        bond_centers:npt.NDArray[np.float64], # [bondcent,3]
                        UNITCELL_VECTOR:np.ndarray,
                        list_atomic_number:list[int] = [6,1,8], # [C,H,O]
                        list_maxat:list[int] = [24,24,24], #  [C,H,O] 
                        Rcs:float =4.0, # in Ang
                        Rc:float  =6.0,  # in Ang
                        device:Literal["cuda","cpu","mps"]="cpu" # "cuda", "cpu" or "mps"
                        )->torch.Tensor:
        if len(bond_centers.shape) != 2 or bond_centers.shape[1] != 3:
            raise ValueError(f"bond_centers should be 2D array. bond_centers.shape should be (bondcent,3) :: {bond_centers.shape}")
        list_mol_coords  = np.array(atomic_coordinate,dtype="float32")
        list_atomic_nums = np.array(atomic_numbers, dtype="int32") # 先にint32に変換しておく必要あり
        # convert numpy array to torch
        list_mol_coords  = torch.tensor(list_mol_coords).to(device)
        list_atomic_nums = torch.tensor(list_atomic_nums).to(device)
        bond_centers     = torch.tensor(np.array(bond_centers,dtype="float32")).to(device)
        # 分子座標-ボンドセンター座標を行列の形で実行する
        # list_mol_coords:: [Frame,]
        # mat_ij = atom_i - atom_
        mat_atom = list_mol_coords[None,:,:].repeat(len(bond_centers),1,1) # 原子座標
        mat_bc   = bond_centers[None,:,:].repeat(len(list_mol_coords),1,1) # ボンドセンター座標
        mat_bc   = torch.transpose(mat_bc, 1,0)
        drs: Float[torch.Tensor, "bondcent atom distance=3"] = (mat_atom - mat_bc) # drs:: [bondcent,Atom,3]

        # apply pbc to drs
        dist_wVec:torch.Tensor = pbc(pbc_3d_torch).compute_pbc(vectors_array = drs,
                                                    cell = UNITCELL_VECTOR,
                                                    device = device)# [bondcent,Atom,3]

        # get atomic numbers from atoms
        # ! CAUTION:: index is different from raw_get_desc_bondcent_allinone
        list_descs = []
        for at,MaxAt in zip(list_atomic_number,list_maxat): # at=6,1,8
            atoms_indx = torch.argwhere(list_atomic_nums==at).reshape(-1) # get index for each atom
            #for C atoms (all) 
            #C原子のローンペアはありえないので原子間距離ゼロの判定は省く
            dist_atoms:torch.Tensor = dist_wVec[:,atoms_indx,:]
            # 距離0の原子を省く．これを入れておけば，lone pairにも対応できる．
            dist_atoms:torch.Tensor = dist_atoms[torch.sum(dist_atoms**2,axis=2)>0.0001].reshape((len(bond_centers),-1,3)) #各行に１つづつ重複した原子が存在するはず
            # calculate generalized coordinate s(r)*(1,x/r,y/r,z/r)
            dij:torch.Tensor = cls.calc_sorted_generalized_coordinate(dist_atoms,Rcs,Rc) # [bondcent,Atom,4]
            # 4d vectorのatomと最後の軸を潰して2次元化する．
            #if len(neighbor list) < MaxAt, zero-padding to MaxAt. 
            dij_descs = cls.fix_length_desc(dij.reshape((len(bond_centers),-1)),MaxAt,device)
            list_descs.append(dij_descs)
        return torch.concatenate(list_descs, axis=1)