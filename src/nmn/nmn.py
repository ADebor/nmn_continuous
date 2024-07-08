import torch as th
from torch import nn, Tensor
from torch.nn import Sequential as Seq
from typing import Optional, Dict, Iterable, Union


class Nmod(nn.Module):

    nmdict = dict()  # will this cause issues?

    def __init__(
        self,
        layers: Seq,
        in_keys: Optional[Iterable] = None,
        out_keys: Optional[Iterable] = None,
        in_nmids: Optional[Iterable] = None,
        out_nmids: Optional[Iterable] = None,
    ):
        """
        Initializes an instance of the Nmod class.

        Args:
            layers (Seq): A sequence of layers.
            in_key (Optional[list[str]], optional): A list of input keys. Defaults to None.
            out_key (Optional[list[str]], optional): A list of output keys. Defaults to None.
            nmids (Optional[list[int]], optional): A list of layers IDs if in_key xor out_key is a list. Defaults to None.
        """
        super(Nmod, self).__init__()
        self.layers = layers
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.in_nmids = in_nmids
        self.out_nmids = out_nmids
        if self.in_keys is not None:

            if self.in_nmids is None:
                self.in_nmids = []
                for id, layer in enumerate(layers):
                    if hasattr(layer, "apply_nm_signal") and callable(
                        layer.apply_nm_signal
                    ):
                        self.in_nmids.append(id)

            assert len(self.in_keys) in [
                len(self.in_nmids),
                1,
            ], "The number of in_keys and nmids must be the same or 1."

            if len(self.in_keys) == 1:
                self.in_keys = len(self.in_nmids) * self.in_keys

            for key, nmid in zip(self.in_keys, self.in_nmids):
                self.layers[nmid].register_forward_pre_hook(
                    self.nm_in_hook(key=key, layer=self.layers[nmid])
                )

        if self.out_keys is not None:

            if self.out_nmids is not None:
                assert len(self.out_keys) == len(
                    self.out_nmids
                ), "The number of out_keys and nmids must be the same."
            else:
                self.out_nmids = len(self.out_keys) * [
                    -1
                ]  # by default, in neuromodulating mode, the neuromodulation signal is the output of the last layer

            for key, nmid in zip(self.out_keys, self.out_nmids):
                self.layers[nmid].register_forward_hook(self.nm_out_hook(key=key))

    def forward(self, x: Tensor):
        for layer in self.layers:  # pay attention to the batch size !!!
            # if isinstance(layer, LSTM):
            #     if not hasattr(layer, "h"):
            #         if layer.batch_first:
            #             layer.h = th.zeros(
            #                 x.shape[1], layer.num_layers, layer.hidden_size
            #             ).to(
            #                 device
            #             )  # hidden state
            #             layer.c = th.zeros(
            #                 x.shape[1], layer.num_layers, layer.hidden_size
            #             ).to(
            #                 device
            #             )  # internal state

            #         else:
            #             layer.h = th.zeros(
            #                 layer.num_layers, x.shape[1], layer.hidden_size
            #             ).to(
            #                 device
            #             )  # hidden state
            #             layer.c = th.zeros(
            #                 layer.num_layers, x.shape[1], layer.hidden_size
            #             ).to(
            #                 device
            #             )  # internal state

            #     x, (layer.h, layer.c) = layer(x, (layer.h, layer.c))
            # else:
            #     x = layer(x)
            x = layer(x)
        return x

    def nm_out_hook(self, key: str):
        def write_nm_signal(_, __, output):
            Nmod.nmdict[key] = output.to(th.float32)

        return write_nm_signal

    def nm_in_hook(self, key: str, layer: nn.Module):
        def apply_nm_signal(cls, input):
            nm_signal = Nmod.nmdict[key]
            nm_input = cls.apply_nm_signal(input[0], nm_signal)
            return nm_input

        return apply_nm_signal


class NmActivation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nm_signal_dim: int,
        device: str,
        *args,
        **kwargs,
    ):
        super(NmActivation, self).__init__()
        self.input_dim = input_dim
        self.nm_signal_dim = nm_signal_dim
        self.device = device

        self.nm_params = self.create_nm_parameters()

    # def activation(self, x):
    #     raise NotImplementedError("This method must be implemented in the child class.")

    def apply_nm_signal(self, input: Tensor, nm_signal: Tensor) -> Tensor:
        return self.neuromodulate(input, nm_signal)

    def neuromodulate(self, input: Tensor, nm_signal: Tensor) -> Tensor:
        raise NotImplementedError("This method must be implemented in the child class.")

    def create_nm_parameters(self, input: Tensor, nm_signal: Tensor) -> None:
        raise NotImplementedError(
            "This method must be implemented in the child class. One must create the nm parameters to be used in self.neuromodulate."
        )

    # def forward(self, x: Tensor):
    #     return self.activation(x)


class VecovenActivation(NmActivation):
    def __init__(self, 
                input_dim: int, 
                nm_signal_dim: int, 
                device: str,
                activation: nn.Module = nn.ReLU(), 
                ):
        super().__init__(input_dim, nm_signal_dim, device)
        self.activation = activation

    # def activation(self, x: Tensor):
    #     return self.activation(x)

    def create_nm_parameters(self):
        w_s = nn.Parameter(
            th.rand((self.nm_signal_dim, self.input_dim)).to(self.device)
        )
        w_b = nn.Parameter(
            th.rand((self.nm_signal_dim, self.input_dim)).to(self.device)
        )

        return nn.ParameterList([w_s, w_b])

    def neuromodulate(self, input: Tensor, nm_signal: Tensor):

        scale = th.matmul(nm_signal, self.nm_params[0])
        bias = th.matmul(nm_signal, self.nm_params[1])
        return scale * input + bias
    
    def forward(self, x:Tensor):
        return self.activation(x)


class GeadahActivation(NmActivation):

    def create_nm_parameters(self):
        w_n = nn.Parameter(
            th.rand((self.nm_signal_dim, self.input_dim)).to(self.device)
        )
        w_s = nn.Parameter(
            th.rand((self.nm_signal_dim, self.input_dim)).to(self.device)
        )

        return nn.ParameterList([w_n, w_s])

    def neuromodulate(self, input: Tensor, nm_signal: Tensor):

        n = th.matmul(nm_signal, self.nm_params[0])
        s = th.matmul(nm_signal, self.nm_params[1])
        exp = th.exp(n * input)
        return (1 - s) * th.log(1 + exp) / n + s * exp / (1 + exp)
    
    def forward(self, x:Tensor):
        return x


class Nmnet(nn.Module):
    def __init__(
        self,
        main_net: Nmod,
        nm_nets: Union[list[Nmod], nn.ModuleList],
        name: str = None,
    ):
        super(Nmnet, self).__init__()
        self.main_net = main_net
        self.nm_nets = nn.ModuleList(nm_nets)
        self.name = name

        # networks names for input dispatching
        self.names = [nm_net.name for nm_net in nm_nets]
        self.names.append(main_net.name)

    def forward(self, x: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        if isinstance(x, Tensor):
            for nm_net in self.nm_nets:
                nm_net(x)

            return self.main_net(x)
        else:
            for nm_net in self.nm_nets:
                nm_net(x[nm_net.name])
            y = self.main_net(x[self.main_net.name])
            return y
