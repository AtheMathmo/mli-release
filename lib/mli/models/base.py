import abc


class BaseModel(abc.ABC):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset_bn(self):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def interpolated_forward(self, x, alpha, state1, state2):
        pass

    @property
    @abc.abstractmethod
    def use_batchnorm(self):
        return False
