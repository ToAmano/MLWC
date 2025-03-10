"""calculate atomic distances
compute_pbc method converts the vectors array using the minimum image convention (mic) for the given cell.
We implemented the strategy design pattern to allow for different implementations of the compute_pbc method.
The pbc class is a context class that holds an instance of a concrete strategy class inheriting from pbc_abstract.
The compute_pbc method calls the compute_pbc method of the concrete strategy class.
This design pattern allows for easy extension of the compute_pbc method by adding new concrete strategy classes.
The pbc_numpy.py and pbc_torch.py files contain concrete strategy classes that implement the compute_pbc method using numpy and torch, respectively.
The pbc_3d class in pbc_numpy.py and pbc_torch.py implements the compute_pbc method for 3D vectors.
The pbc_2d class in pbc_numpy.py and pbc_torch.py implements the compute_pbc method for 2D vectors.
The pbc_1d class in pbc_numpy.py and pbc_torch.py implements the compute_pbc method for 1D vectors.
The pbc_abstract class defines the interface for the compute_pbc method.
"""

import abc

class pbc_abstract(abc.ABC):
    """
    アルゴリズム（ConcreteStrategy）が実装する共通のインターフェイス
    """
    @classmethod
    @abc.abstractmethod
    def compute_pbc(cls):
        """
        Abstract method for computing periodic boundary conditions.

        Example:
        >>> pbc_abstract.compute_pbc()
        """
        pass
    
class pbc():
    """
    ConcreteStrategy をインスタンス変数として持つクラス
    """
    def __init__(self, strategy): # pbc_abstract
        """
        Initializes the pbc class with a strategy.

        Args:
            strategy: An instance of a concrete strategy class inheriting from pbc_abstract.

        Example:
        >>> from src.cpmd.pbc.pbc_numpy import pbc_3d
        >>> strategy = pbc_3d()
        >>> pbc_instance = pbc(strategy)
        """
        self.strategy = strategy

    def compute_pbc(self,**kwargs):
        """
        Computes periodic boundary conditions using the given strategy.

        Args:
            **kwargs: Keyword arguments to be passed to the strategy's compute_pbc method.

        Returns:
            The result of the strategy's compute_pbc method.

        Example:
        >>> from src.cpmd.pbc.pbc_numpy import pbc_3d
        >>> strategy = pbc_3d()
        >>> pbc_instance = pbc(strategy)
        >>> vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> result = pbc_instance.compute_pbc(vectors_array=vectors, cell=cell)
        >>> print(result)
        [[ 1.  0.  0.]
         [ 0.  1.  0.]
         [ 0.  0.  1.]]
        """
        # Call ConcreteStrategy method to consignment processing
        return self.strategy.compute_pbc(**kwargs)




