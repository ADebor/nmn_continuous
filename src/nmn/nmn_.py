import torch as th
from torch import Tensor, nn
from torch.nn import Sequential as Seq
from typing import Optional, Dict, Iterable, Union

from torchviz import make_dot
from torch.nn.modules.rnn import RNNBase, LSTM, GRU


class SReLU(nn.ReLU):
    def forward(self, x):
        return th.clamp(super().forward(x + 1.0) - 1.0, max=1.0)


class Nmod(nn.Module):

    nmdict = dict()  # will this cause issues?

    def __init__(
        self,
        layers: Seq,
        in_keys: Optional[Iterable] = None,
        out_keys: Optional[Iterable] = None,
        in_nmids: Optional[Iterable] = None,
        out_nmids: Optional[Iterable] = None,
        # name: Optional[str] = None,
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
        # self.name = name

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
            if isinstance(layer, LSTM):
                if not hasattr(layer, "h"):
                    if layer.batch_first:
                        layer.h = th.zeros(
                            x.shape[1], layer.num_layers, layer.hidden_size
                        ).to(
                            device
                        )  # hidden state
                        layer.c = th.zeros(
                            x.shape[1], layer.num_layers, layer.hidden_size
                        ).to(
                            device
                        )  # internal state

                    else:
                        layer.h = th.zeros(
                            layer.num_layers, x.shape[1], layer.hidden_size
                        ).to(
                            device
                        )  # hidden state
                        layer.c = th.zeros(
                            layer.num_layers, x.shape[1], layer.hidden_size
                        ).to(
                            device
                        )  # internal state

                x, (layer.h, layer.c) = layer(x, (layer.h, layer.c))
            else:
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
    ):
        super(NmActivation, self).__init__()
        self.input_dim = input_dim
        self.nm_signal_dim = nm_signal_dim
        self.device = device

        self.nm_params = self.create_nm_parameters()

    def activation(self, x):
        raise NotImplementedError("This method must be implemented in the child class.")

    def apply_nm_signal(self, input: Tensor, nm_signal: Tensor) -> Tensor:

        # if not hasattr(self, "nm_params"):
        #     self.nm_params = self.create_nm_parameters(input, nm_signal)
        return self.neuromodulate(input, nm_signal)

    def neuromodulate(self, input: Tensor, nm_signal: Tensor) -> Tensor:
        raise NotImplementedError("This method must be implemented in the child class.")

    def create_nm_parameters(self, input: Tensor, nm_signal: Tensor) -> None:
        raise NotImplementedError(
            "This method must be implemented in the child class. One must create the nm parameters to be used in self.neuromodulate."
        )

    def forward(self, x: Tensor):
        return self.activation(x)


class VecovenActivation(NmActivation):
    def __init__(self, activation_fn: nn.Module = nn.ReLU(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation_fn = activation_fn

    def activation(self, x: Tensor):
        return self.activation_fn(x)

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


class GeadahActivation(NmActivation):

    def apply_nm_signal(self, input: Tensor, nm_signal: Tensor):
        scale = th.matmul(nm_signal, self.w_s)
        bias = th.matmul(nm_signal, self.w_b)
        return th.sigmoid(scale * input + bias)


class Nmnet(nn.Module):  # useless?
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


if __name__ == "__main__":

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    main_net = Nmod(
        Seq(
            nn.Linear(10, 32),
            VecovenActivation(nn.ReLU()),
            nn.Linear(32, 32),
            VecovenActivation(nn.Tanh()),
        ),
        in_keys=["nm1"],
    ).to(device)

    nm_net_1 = Nmod(
        Seq(nn.LSTM(10, 64, 2), nn.Linear(64, 32), nn.Tanh()),
        out_keys=["nm1"],
    ).to(device)

    nm_net_2 = Nmod(
        Seq(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU()),
        out_keys=["nm2"],
    ).to(device)

    nmnet = Nmnet(main_net, nn.ModuleList([nm_net_1])).to(device)

    batch_size = 5
    in_dim = 10
    for step in range(3):
        x = step * th.ones([1, batch_size, in_dim], dtype=th.float32, device=device)
        y = nmnet(x)

        dot = make_dot(
            y, params=dict(nmnet.named_parameters()), show_attrs=True, show_saved=True
        )
        dot.format = "png"
        dot.render(f"torchviz-sample-{step}")

    # I dont think having a global "nmnet" class is a good idea, since it would require some
    # init method parameters specifying the neuromodulating and neuromodulated nets
    # and the keys to be used for the neuromodulation signal. I'd rather have a Nmlayer
    # class which imo would be more modular and flexible. The only advantage I see in considering
    # a nm-net (in comparison to a nmlayer, this is diff from the global nmnet class of the beginning
    # of this note) class rather than a nmlayer is that in the case of whole network modulation, it would
    # require to specify neuromodulation specs for each layer and instanciating a nmlayer for each of them.
    # But I guess we could have a nmlayer class with a list of layers to be neuromodulated, and then in the case
    # of diff neuromodulating nets or things a bit more special, then instanciating multiple nmlayers would be inevitable.
    # The only thing that bothers me is that the 'user' would have to create some nmdict (tensordict?) so that signals are shared (edit: no need).

    # idea: class called before being actually instanced -> use of a generator then to instciante (some staticmethod with a yield)
