"""
This module implements a torch-based descriptor calculation for bond center and lone pair environments.
It utilizes periodic boundary conditions (PBC) and cutoff functions to generate generalized coordinates
that can be used as descriptors in machine learning models.

The main class, `Descriptor_torch_bondcenter`, inherits from `Descriptor_abstract` and provides methods
for calculating sorted generalized coordinates, fixing the length of descriptors, and computing the final descriptor.

Examples:
    >>> import ase
    >>> import numpy as np
    >>> import torch
    >>> from mlwc.ml.descriptor.descriptor_torch import Descriptor_torch_bondcenter
    >>> from ase.build import molecule
    >>> atoms = molecule('H2O')
    >>> bond_centers = np.array([[0.0, 0.0, 0.0]])
    >>> descriptor = Descriptor_torch_bondcenter.calc_descriptor(atoms, bond_centers)
    >>> print(descriptor.shape)
    (1, 288)
"""

from typing import List, Literal, Optional

import ase
import numpy as np
import numpy.typing as npt  # for annotation
import torch
import torch.nn

# from torchtyping import TensorType # for annotation
from jaxtyping import Float

from mlwc.cpmd.descripter import cutoff_func_torch
from mlwc.cpmd.pbc.pbc import pbc
from mlwc.cpmd.pbc.pbc_torch import pbc_3d_torch
from mlwc.ml.descriptor.descriptor_abstract import DescriptorAbstract


class DescriptorTorchBondcenter(DescriptorAbstract):
    """
    Calculates descriptors for bond center and lone pair environments using torch.

    This class inherits from `Descriptor_abstract` and implements methods for calculating
    descriptors based on the distances and cutoff functions applied to neighboring atoms
    around bond centers. It utilizes torch tensors for efficient computation and supports
    periodic boundåœary conditions.

    Attributes
    ----------
    None

    Methods
    -------
    calc_sorted_generalized_coordinate(distance_array_3d, Rcs, Rc)
        Sorts distances and applies a cutoff function to generate generalized coordinates.
    fix_length_desc(desc, MaxAt, device)
        Fixes the length of the descriptor by padding or truncating it.
    calc_descriptor(atoms, bond_centers, list_atomic_number, list_maxat, Rcs, Rc, device)
        Calculates the descriptor for a given set of atoms and bond centers.
    forward(atomic_coordinate, atomic_numbers, bond_centers, UNITCELL_VECTOR, list_atomic_number, list_maxat, Rcs, Rc, device)
        Calculates the descriptor for a given set of atomic coordinates, atomic numbers, and bond centers.

    Examples
    --------
    >>> import ase
    >>> import numpy as np
    >>> import torch
    >>> from mlwc.ml.descriptor.descriptor_torch import Descriptor_torch_bondcenter
    >>> from ase.build import molecule
    >>> atoms = molecule('H2O')
    >>> bond_centers = np.array([[0.0, 0.0, 0.0]])
    >>> descriptor = Descriptor_torch_bondcenter.calc_descriptor(atoms, bond_centers)
    >>> print(descriptor.shape)
    (1, 288)
    """

    def __init__(self):
        super().__init__()
        self.pbc_torch = pbc_3d_torch()

    @torch.jit.export
    def calc_sorted_generalized_coordinate(
        self,
        # Float[torch.Tensor, "bondcent atom distance"],
        distance_array_3d: torch.Tensor,
        Rcs: float,
        Rc: float,
    ) -> torch.Tensor:  # Float[torch.Tensor, "bondcent atom distance+1"]:  # distnace:4
        """
        Sorts distances and applies a cutoff function to generate generalized coordinates.

        This method calculates the distances between bond centers and neighboring atoms,
        applies a cutoff function to these distances, and then sorts the distances based on
        the cutoff function values. The sorted distances and cutoff function values are
        then used to generate generalized coordinates.

        Parameters
        ----------
        distance_array_3d : Float[torch.Tensor, "bondcent atom distance"]
            A 3D tensor containing the distances between bond centers and neighboring atoms.
        Rcs : float
            The cutoff radius for the short-range interaction.
        Rc : float
            The cutoff radius for the long-range interaction.

        Returns
        -------
        Float[torch.Tensor, "bondcent atom distance+1"]
            A tensor containing the sorted distances and cutoff function values.

        Examples
        --------
        >>> import torch
        >>> distance_array_3d = torch.randn(1, 3, 3)
        >>> Rcs = 4.0
        >>> Rc = 6.0
        >>> dij = Descriptor_torch_bondcenter.calc_sorted_generalized_coordinate(distance_array_3d, Rcs, Rc)
        >>> print(dij.shape)
        torch.Size([1, 3, 4])
        """

        # r = torch.linalg.norm(distances, dim=-1)
        # s_r = cutoff_func_torch(r, rcs, rc)
        # scaled_vecs = s_r.unsqueeze(-1) * distances / r.unsqueeze(-1)
        # features = torch.cat([s_r.unsqueeze(-1), scaled_vecs], dim=-1)

        # sorted_idx = torch.argsort(s_r, dim=1, descending=True)
        # return torch.gather(features, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 4))

        d_r = torch.sqrt(
            torch.sum(distance_array_3d**2, dim=2)
        )  # calculate distance from sqrt(x^2+y^2+z^2)
        s_r = cutoff_func_torch(d_r, Rcs, Rc)
        # (s(r)*x/r, s(r)*y/r, s(r)*z/r)
        scaled_3dvec = s_r[:, :, None] * distance_array_3d / d_r[:, :, None]
        # s(r), s(r)*x/r, s(r)*y/r, s(r)*z/r
        dij = torch.cat([s_r[:, :, None], scaled_3dvec], dim=2)
        # sort on num_atom with 1/r
        sorted_indices = torch.argsort(
            s_r, dim=1, descending=True
        )  # s_r はdij[..., 0]でもok
        return torch.gather(
            dij, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 4)
        )  # sort tensor

    @torch.jit.export
    def fix_length_desc(
        self, desc: torch.Tensor, MaxAt: int, device: str
    ) -> torch.Tensor:
        """
        Fixes the length of the descriptor by padding or truncating it.

        This method ensures that the descriptor has a fixed length by either padding it with zeros
        if it is shorter than the maximum allowed length (`MaxAt * 4`) or truncating it if it is longer.

        Parameters
        ----------
        desc : torch.Tensor
            The descriptor tensor.
        MaxAt : int
            The maximum number of atoms to consider.
        device : str
            The device to use for the padding tensor (e.g., "cpu" or "cuda").

        Returns
        -------
        torch.Tensor
            The fixed-length descriptor tensor.

        Examples
        --------
        >>> import torch
        >>> desc = torch.randn(1, 10)
        >>> MaxAt = 5
        >>> device = "cpu"
        >>> fixed_desc = Descriptor_torch_bondcenter.fix_length_desc(desc, MaxAt, device)
        >>> print(fixed_desc.shape)
        torch.Size([1, 20])
        """
        expected_len = MaxAt * 4
        current_len = desc.size(1)
        if current_len < expected_len:
            padding = torch.zeros(desc.size(0), expected_len - current_len).to(device)
            desc = torch.cat([desc, padding], dim=1)
        return desc[:, :expected_len]

    def calc_descriptor(
        self,
        atoms: ase.Atoms,
        bond_centers: npt.NDArray[np.float64],  # [bondcent,3]
        list_atomic_number: list[int] = [6, 1, 8],  # [C,H,O]
        list_maxat: list[int] = [24, 24, 24],  # [C,H,O]
        Rcs: float = 4.0,  # in Ang
        Rc: float = 6.0,  # in Ang
        device: str = "cpu",  # "cuda", "cpu" or "mps"
    ) -> np.ndarray:
        """
        Calculates the descriptor for a given set of atoms and bond centers using numpy.

        This method computes the descriptor based on the atomic positions, bond centers,
        and specified parameters such as cutoff radii and maximum number of atoms.
        It utilizes periodic boundary conditions (PBC) to account for the periodicity of the system.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object containing the atomic positions and cell information.
        bond_centers : npt.NDArray[np.float64]
            A 2D array containing the coordinates of the bond centers.
        list_atomic_number : list[int], optional
            A list of atomic numbers to consider (default is [6, 1, 8] for C, H, and O).
        list_maxat : list[int], optional
            A list of maximum number of atoms to consider for each atomic number (default is [24, 24, 24]).
        Rcs : float, optional
            The cutoff radius for the short-range interaction (default is 4.0 Angstrom).
        Rc : float, optional
            The cutoff radius for the long-range interaction (default is 6.0 Angstrom).
        device : str, optional
            The device to use for torch tensors (default is "cpu").

        Returns
        -------
        np.ndarray
            The calculated descriptor as a NumPy array.

        Raises
        ------
        ValueError
            If the shape of `bond_centers` is not (bondcent, 3).

        Examples
        --------
        >>> import ase
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> from mlwc.ml.descriptor.descriptor_torch import Descriptor_torch_bondcenter
        >>> atoms = molecule('H2O')
        >>> bond_centers = np.array([[0.0, 0.0, 0.0]])
        >>> descriptor = Descriptor_torch_bondcenter.calc_descriptor(atoms, bond_centers)
        >>> print(descriptor.shape)
        (1, 288)
        """
        if len(bond_centers.shape) != 2 or bond_centers.shape[1] != 3:
            raise ValueError(
                f"bond_centers should be 2D array. bond_centers.shape should be (bondcent,3) :: {bond_centers.shape}"
            )
        list_mol_coords = np.array(atoms.get_positions(), dtype="float32")
        list_atomic_nums = np.array(
            atoms.get_atomic_numbers(), dtype="int32"
        )  # 先にint32に変換しておく必要あり
        # convert numpy array to torch
        list_mol_coords = torch.tensor(list_mol_coords).to(device)
        list_atomic_nums = torch.tensor(list_atomic_nums).to(device)
        bond_centers = torch.tensor(np.array(bond_centers, dtype="float32")).to(device)
        # 分子座標-ボンドセンター座標を行列の形で実行する
        # list_mol_coords:: [Frame,]
        # mat_ij = atom_i - atom_
        mat_atom = list_mol_coords[None, :, :].repeat(
            len(bond_centers), 1, 1
        )  # 原子座標
        mat_bc = bond_centers[None, :, :].repeat(
            len(list_mol_coords), 1, 1
        )  # ボンドセンター座標
        mat_bc = torch.transpose(mat_bc, 1, 0)
        drs: Float[torch.Tensor, "bondcent atom distance=3"] = (
            mat_atom - mat_bc
        )  # drs:: [bondcent,Atom,3]

        # apply pbc to drs
        dist_wVec: torch.Tensor = pbc(pbc_3d_torch).compute_pbc(
            vectors_array=drs, cell=np.array(atoms.get_cell()), device=device
        )  # [bondcent,Atom,3]

        # get atomic numbers from atoms
        # ! CAUTION:: index is different from raw_get_desc_bondcent_allinone
        list_descs = []
        for at, MaxAt in zip(list_atomic_number, list_maxat):  # at=6,1,8
            # get index for each atom
            atoms_indx = torch.argwhere(list_atomic_nums == at).reshape(-1)
            # for C atoms (all)
            # C原子のローンペアはありえないので原子間距離ゼロの判定は省く
            dist_atoms: torch.Tensor = dist_wVec[:, atoms_indx, :]
            # 距離0の原子を省く．これを入れておけば，lone pairにも対応できる．
            dist_atoms: torch.Tensor = dist_atoms[
                torch.sum(dist_atoms**2, dim=2) > 0.0001
            ].reshape(
                (len(bond_centers), -1, 3)
            )  # 各行に１つづつ重複した原子が存在するはず
            # calculate generalized coordinate s(r)*(1,x/r,y/r,z/r)
            dij: torch.Tensor = self.calc_sorted_generalized_coordinate(
                dist_atoms, Rcs, Rc
            )  # [bondcent,Atom,4]
            # if len(neighbor list) < MaxAt, zero-padding to MaxAt.
            dij_descs = self.fix_length_desc(
                dij.reshape((len(bond_centers), -1)), MaxAt, device
            )
            list_descs.append(dij_descs)
        return np.concatenate(list_descs, axis=1)

    def forward(
        self,
        list_mol_coords: torch.Tensor,
        list_atomic_nums: torch.Tensor,
        bond_centers: torch.Tensor,  # [bondcent,3]
        UNITCELL_VECTOR: torch.Tensor,
        list_atomic_number: Optional[torch.Tensor] = None,  # [C,H,O],[6, 1, 8]
        list_maxat: Optional[torch.Tensor] = None,  # [C,H,O],[24, 24, 24],
        Rcs: float = 4.0,  # in Ang
        Rc: float = 6.0,  # in Ang
        device: str = "cpu",  # "cuda", "cpu" or "mps"
    ) -> torch.Tensor:
        """
        Calculates the descriptor for a given set of atomic coordinates, atomic numbers, and bond centers using PyTorch.

        This method computes the descriptor based on the atomic coordinates, atomic numbers, bond centers,
        unit cell vector, and specified parameters such as cutoff radii and maximum number of atoms.
        It utilizes periodic boundary conditions (PBC) to account for the periodicity of the system.

        Parameters
        ----------
        atomic_coordinate : np.ndarray
            A NumPy array containing the atomic coordinates.
        atomic_numbers : np.ndarray
            A NumPy array containing the atomic numbers.
        bond_centers : npt.NDArray[np.float64]
            A 2D array containing the coordinates of the bond centers.
        UNITCELL_VECTOR : np.ndarray
            A NumPy array representing the unit cell vector.
        list_atomic_number : list[int], optional
            A list of atomic numbers to consider (default is [6, 1, 8] for C, H, and O).
        list_maxat : list[int], optional
            A list of maximum number of atoms to consider for each atomic number (default is [24, 24, 24]).
        Rcs : float, optional
            The cutoff radius for the short-range interaction (default is 4.0 Angstrom).
        Rc : float, optional
            The cutoff radius for the long-range interaction (default is 6.0 Angstrom).
        device : Literal["cuda", "cpu", "mps"], optional
            The device to use for torch tensors (default is "cpu").

        Returns
        -------
        torch.Tensor
            The calculated descriptor as a torch tensor.

        Raises
        ------
        ValueError
            If the shape of `bond_centers` is not (bondcent, 3).

        Examples
        --------
        >>> import ase
        >>> import numpy as np
        >>> import torch
        >>> from ase.build import molecule
        >>> from mlwc.ml.descriptor.descriptor_torch import Descriptor_torch_bondcenter
        >>> atoms = molecule('H2O')
        >>> atomic_coordinate = atoms.get_positions()
        >>> atomic_numbers = atoms.get_atomic_numbers()
        >>> bond_centers = np.array([[0.0, 0.0, 0.0]])
        >>> UNITCELL_VECTOR = atoms.get_cell()
        >>> descriptor = Descriptor_torch_bondcenter.forward(atomic_coordinate, atomic_numbers, bond_centers, UNITCELL_VECTOR)
        >>> print(descriptor.shape)
        torch.Size([1, 288])
        """
        if len(bond_centers.shape) != 2 or bond_centers.shape[1] != 3:
            raise ValueError(
                f"bond_centers should be 2D array. bond_centers.shape should be (bondcent,3) :: {bond_centers.shape}"
            )
        if device not in ["cpu", "cuda", "mps"]:
            raise ValueError(
                f"deice should be one of cpu, cuda, and mps :: got {device}"
            )
        if list_atomic_number is None:
            list_atomic_number = torch.tensor([6, 1, 8], dtype=torch.int, device=device)
        if list_maxat is None:
            list_maxat = torch.tensor([24, 24, 24], dtype=torch.int, device=device)
        if list_atomic_number.dim() != 1 or list_maxat.dim() != 1:
            raise ValueError(
                f"list_atomic_number and list_maxat should be 1D array. :: {list_atomic_number.dim()}"
            )

        # 分子座標-ボンドセンター座標を行列の形で実行する
        # list_mol_coords:: [Frame,]
        # mat_ij = atom_i - atom_
        mat_atomic_coord = list_mol_coords[None, :, :].repeat(len(bond_centers), 1, 1)
        mat_bc_coord = bond_centers[None, :, :].repeat(len(list_mol_coords), 1, 1)
        mat_bc_coord = torch.transpose(mat_bc_coord, 1, 0)
        # drs: Float[torch.Tensor, "bondcent atom distance=3"] = (
        #     mat_atom - mat_bc)  # drs:: [bondcent,Atom,3]
        drs = mat_atomic_coord - mat_bc_coord  # drs:: [bondcent,Atom,3]

        # apply pbc to drs
        distance_atom_bc: torch.Tensor = self.pbc_torch.forward(
            vectors_array=drs, cell=UNITCELL_VECTOR
        )  # [bondcent,Atom,3]
        # get atomic numbers from atoms
        # ! CAUTION:: index is different from raw_get_desc_bondcent_allinone
        list_descs = []
        # for at, MaxAt in zip(list_atomic_number, list_maxat):  # at=6,1,8
        for index in range(len(list_atomic_number)):
            at = list_atomic_number[index]
            MaxAt = list_maxat[index]
            # get index for each atom
            atoms_indx = torch.argwhere(list_atomic_nums == at).reshape(-1)
            dist_atoms: torch.Tensor = distance_atom_bc[:, atoms_indx, :]
            # 距離0の原子を省く．これを入れておけば，lone pairにも対応できる．また，各行に１つづつ重複した原子が存在するはず
            dist_atoms: torch.Tensor = dist_atoms[
                torch.sum(dist_atoms**2, dim=2) > 0.0001
            ].reshape((len(bond_centers), -1, 3))
            # calculate generalized coordinate s(r)*(1,x/r,y/r,z/r)
            dij: torch.Tensor = self.calc_sorted_generalized_coordinate(
                dist_atoms, Rcs, Rc
            )  # [bondcent,Atom,4]
            # 4d vectorのatomと最後の軸を潰して2次元化する．
            # if len(neighbor list) < MaxAt, zero-padding to MaxAt.
            dij_descs = self.fix_length_desc(
                dij.reshape((len(bond_centers), -1)), MaxAt, device
            )
            list_descs.append(dij_descs)
        descs = torch.cat(list_descs, dim=1)
        # print(descs.size())
        # print(f"descs = {descs[:20]}")
        return descs
