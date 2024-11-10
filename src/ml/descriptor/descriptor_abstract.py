from abc import (
    ABC,
    abstractmethod,
)
import __version__

class Descriptor_abstract(ABC):
    """abstract method for dataset

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        self.version = __version__.__version__  # use version info for backward compatibility
        super().__init__()
        
        pass
    
    @abstractmethod
    def calc_descriptor(self):
        pass

class Descriptor():
    """
    ConcreteStrategy をインスタンス変数として持つクラス
    """
    def __init__(self, strategy: type[Descriptor_abstract]):
        self._strategy = strategy

    def calc_descriptor(self,**kwargs):
        # ConcreteStrategy のメソッドを呼ぶことで、一部の処理を委託する
        return self._strategy.calc_descriptor(**kwargs)
        
    @property
    def strategy(self) -> Descriptor_abstract:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """
        return self._strategy
    
    
    @strategy.setter
    def strategy(self, strategy: Descriptor_abstract) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """
        self._strategy = strategy