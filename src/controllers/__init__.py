REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
