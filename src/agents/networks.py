import torch as th
from torch import nn
from typing import Dict, Tuple, Union, List, Optional, Type
from nmn import Nmod
import nmn
from omegaconf import ListConfig
from nmn import activation_filter

class AcNetwork(nn.Module):
    def __init__(
        self,
        features_dim: Union[int, str],
        layers: Optional[Union[List[int], ListConfig, Dict]] = [20, 10],
        activation_cls: Type[nn.Module] = nn.ReLU,
        device: th.device = th.device("cuda:0" if th.cuda.is_available() else "cpu"),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.features_dim = features_dim
        self.latent_dim_pi = (
            layers[-1] if type(layers) in [list, ListConfig] else layers["pi"][-1]
        )
        self.latent_dim_vf = (
            layers[-1] if type(layers) in [list, ListConfig] else layers["vf"][-1]
        )
        self.device = device
        self.layers = layers
        self.activation_cls = activation_cls

        self.setup_ac_networks()

    @staticmethod
    def fill_sequential(
        layers: List[int],
        in_size: int,
        activation_cls: Type[nn.Module],
        activation_kwargs: Dict = {},
    ) -> nn.Module:
        sequential = nn.Sequential()
        for i, layer_size in enumerate(layers):
            sequential.add_module(f"layer_{i}", nn.Linear(in_size, layer_size))
            sequential.add_module(
                f"activation_{i}",
                activation_filter(
                    activation_cls,
                    input_dim=layer_size, 
                    **activation_kwargs,
                )
            )
            in_size = layer_size
        return sequential

    def setup_ac_networks(self) -> None:
        self.policy_net, self.value_net = (
            self.fill_sequential(
                self.layers, self.features_dim, self.activation_cls
            ),
            self.fill_sequential(
                self.layers, self.features_dim, self.activation_cls
            ),
        )

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class NmnAcNetwork(AcNetwork):
    def __init__(
        self,
        rnn_features_dim: int,
        nm_signal_dim: int,
        nm_layers: Optional[Union[List[int], Dict]] = [30, 10],
        nm_activation_cls: Type[nmn.NmActivation] = nmn.VecovenActivation,
        **kwargs,
    ) -> None:
        self.nm_signal_dim = nm_signal_dim
        self.rnn_features_dim = rnn_features_dim
        self.nm_layers = nm_layers
        self.nm_activation_cls = nm_activation_cls
        super().__init__(**kwargs)

    def setup_ac_networks(self) -> None:
        policy_main_net, value_main_net = nn.Sequential(), nn.Sequential()
        policy_nm_net, value_nm_net = nn.Sequential(), nn.Sequential()

        policy_main_net, value_main_net = self.fill_sequential(
            self.layers, self.features_dim, self.activation_cls
        ), self.fill_sequential(self.layers, self.features_dim, self.activation_cls)

        policy_nm_net, value_nm_net = (
            self.fill_sequential(
                self.nm_layers,
                self.rnn_features_dim,
                self.nm_activation_cls,
                activation_kwargs={
                    "activation": self.activation_cls(),
                    "nm_signal_dim": self.nm_signal_dim,
                    "device": self.device,
                },
            ),
            self.fill_sequential(
                self.nm_layers,
                self.rnn_features_dim,
                self.nm_activation_cls,
                activation_kwargs={
                    "activation": self.activation_cls(),
                    "nm_signal_dim": self.nm_signal_dim,
                    "device": self.device,
                },
            ),
        )

        policy_main_net = Nmod(
            policy_main_net,
            in_keys=["nm_z_actor"],
        ).to(self.device)
        value_main_net = Nmod(
            value_main_net,
            in_keys=["nm_z_critic"],
        ).to(self.device)

        policy_nm_net = Nmod(
            policy_nm_net,
            out_keys=["nm_z_actor"],
        ).to(self.device)
        value_nm_net = Nmod(
            value_nm_net,
            out_keys=["nm_z_critic"],
        ).to(self.device)

        self.policy_net = nn.ModuleDict(
            {"main_net_actor": policy_main_net, "nm_net_actor": policy_nm_net}
        )
        self.value_net = nn.ModuleDict(
            {"main_net_critic": value_main_net, "nm_net_critic": value_nm_net}
        )

    def forward_actor(self, x: Union[th.Tensor, Dict[str, th.Tensor]]) -> th.Tensor:
        # dispatch input to the corresponding network
        if isinstance(x, Dict):
            x = dict(nm_net_actor=x["rnn_output"], main_net_actor=x["fe_output"])

        # neuromodulating network forward pass
        _ = self.policy_net["nm_net_actor"](x["nm_net_actor"])

        # neuromodulated network forward pass
        latent_pi = self.policy_net["main_net_actor"](x["main_net_actor"])

        return latent_pi

    def forward_critic(self, x: Union[th.Tensor, Dict[str, th.Tensor]]) -> th.Tensor:
        # dispatch input to the corresponding network
        if isinstance(x, Dict):
            x = dict(nm_net_critic=x["rnn_output"], main_net_critic=x["fe_output"])

        # neuromodulating network forward pass
        _ = self.value_net["nm_net_critic"](x["nm_net_critic"])

        # neuromodulated network forward pass
        latent_vf = self.value_net["main_net_critic"](x["main_net_critic"])

        return latent_vf
