import torch as th
from torch import nn
import inspect

def activation_filter(activation_class, *args, **kwargs):
    # Get the signature of the activation class
    sig = inspect.signature(activation_class)
    # Filter kwargs to only those accepted by the activation_class
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Return an instance of the activation class with the filtered kwargs
    return activation_class(*args, **filtered_kwargs)

# class ReLU(nn.ReLU):
#     def __init__(self, inplace: bool = False, *args, **kwargs):
#         super().__init__(inplace)

class SReLU(nn.ReLU):
    def forward(self, x):
        return th.clamp(super().forward(x + 1.0) - 1.0, max=1.0)


# quick fix to handle tupled hidden state
class GRU(nn.GRU):
    def forward(self, input, hx):
        output, hx = super().forward(input, hx[0])
        return output, (hx,)
