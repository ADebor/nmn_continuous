import numpy as np
from typing import Tuple, Dict, Type
from functools import partial
import torch as th
from torch import nn
from torch.nn import Sequential as Seq

from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    BetaDistribution,
)
from stable_baselines3.common.type_aliases import Schedule

from nmn import Nmod, VecovenActivation
from utils import get_class_from_path


class CustomMlpRecurrentAcPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        mlp_cls: str,
        mlp_kwargs: Dict[str, any],
        *args,
        **kwargs,
    ):
        self.mlp_cls = get_class_from_path(mlp_cls)
        self.mlp_kwargs = mlp_kwargs
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self.mlp_cls(**self.mlp_kwargs)


class NmnRecurrentAcPolicy(CustomMlpRecurrentAcPolicy):
    def __init__(
        self,
        is_action_net_nmd=True,
        is_value_net_nmd=True,
        nm_activation_cls: Type[nn.Module] = VecovenActivation,
        *args,
        **kwargs,
    ):
        self.is_action_net_nmd = is_action_net_nmd
        self.is_value_net_nmd = is_value_net_nmd
        self.nm_activation_cls = nm_activation_cls

        super().__init__(*args, **kwargs)
        
    def extract_meta_obs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        prev_obs = obs[:, self.observation_space.init_obs_dim :]
        obs = obs[:, : self.observation_space.init_obs_dim]
        return obs, prev_obs

    def forward(
        self,
        obs: th.Tensor,
        rnn_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # Preprocess the observation if needed
        # features = self.extract_features(obs)   # useless

        # if self.share_features_extractor:
        #     pi_features = vf_features = features  # alis
        # else:
        #     pi_features, vf_features = features
        # obs, prev_obs = obs
        obs, prev_obs = self.extract_meta_obs(obs)

        # RNN processing
        rnn_features = prev_obs
        latent_pi, rnn_states_pi = self._process_sequence(
            rnn_features, rnn_states.pi, episode_starts, self.rnn_actor
        )
        if self.rnn_critic is not None:
            latent_vf, rnn_states_vf = self._process_sequence(
                rnn_features, rnn_states.vf, episode_starts, self.rnn_critic
            )
        elif self.shared_rnn:
            # Re-use rnn features but do not backpropagate
            latent_vf = latent_pi.detach()
            if self.rnn_type == "lstm":
                rnn_states_vf = (rnn_states_pi[0].detach(), rnn_states_pi[1].detach())
            else:
                rnn_states_vf = (rnn_states_pi[0].detach(),)
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(
                rnn_features
            )  # should maybe be features instead of rnn_features
            rnn_states_vf = rnn_states_pi

        # "MLP" extractors
        mlp_features = obs
        latent_pi = self.mlp_extractor.forward_actor(
            dict(
                rnn_output=latent_pi,
                fe_output=mlp_features,
            )
        )
        latent_vf = self.mlp_extractor.forward_critic(
            dict(
                rnn_output=latent_vf,
                fe_output=mlp_features,
            )
        )

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(rnn_states_pi, rnn_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        rnn_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        # features = self.extract_features(obs, self.pi_features_extractor)   # useless
        obs, prev_obs = self.extract_meta_obs(obs)

        # RNN processing
        rnn_features = prev_obs
        latent_pi, rnn_states = self._process_sequence(
            rnn_features, rnn_states, episode_starts, self.rnn_actor
        )

        # "MLP" extractors
        mlp_features = obs
        latent_pi = self.mlp_extractor.forward_actor(
            dict(
                rnn_output=latent_pi,
                fe_output=mlp_features,
            )
        )
        return self._get_action_dist_from_latent(latent_pi), rnn_states

    def predict_values(
        self,
        obs: th.Tensor,
        rnn_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :return: the estimated values.
        """
        # NOTE: This is the same logic as in the forward method.

        # Call the method from the parent of the parent class
        # features = self.extract_features(obs, self.vf_features_extractor)   # useless
        obs, prev_obs = self.extract_meta_obs(obs)

        # RNN processing
        rnn_features = prev_obs
        if self.rnn_critic is not None:
            latent_vf, rnn_states_vf = self._process_sequence(
                rnn_features, rnn_states, episode_starts, self.rnn_critic
            )
        elif self.shared_rnn:
            # Use RNN from the actor
            latent_pi, _ = self._process_sequence(
                rnn_features, rnn_states, episode_starts, self.rnn_actor
            )
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(
                rnn_features
            )  # should maybe be features instead of rnn_features

        # "MLP" extractors
        mlp_features = obs
        latent_vf = self.mlp_extractor.forward_critic(
            dict(
                rnn_output=latent_vf,
                fe_output=mlp_features,
            )
        )

        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        rnn_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        # features = self.extract_features(obs)       # useless

        # if self.share_features_extractor:
        #     pi_features = vf_features = features  # alias
        # else:
        #     pi_features, vf_features = features
        obs, prev_obs = self.extract_meta_obs(obs)

        # RNN processing
        rnn_features = prev_obs
        latent_pi, _ = self._process_sequence(
            rnn_features, rnn_states.pi, episode_starts, self.rnn_actor
        )
        if self.rnn_critic is not None:
            latent_vf, _ = self._process_sequence(
                rnn_features, rnn_states.vf, episode_starts, self.rnn_critic
            )
        elif self.shared_rnn:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(rnn_features)

        # "MLP" extractors
        mlp_features = obs
        latent_pi = self.mlp_extractor.forward_actor(
            dict(
                rnn_output=latent_pi,
                fe_output=mlp_features,
            )
        )
        latent_vf = self.mlp_extractor.forward_critic(
            dict(
                rnn_output=latent_vf,
                fe_output=mlp_features,
            )
        )

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def _build_action_value_networks(self) -> None:
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        elif isinstance(self.action_dist, BetaDistribution):
            action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.is_action_net_nmd:
            # terrible hack to make the last two layers nmod
            self.action_net = Nmod(
                Seq(
                    action_net,
                    self.nm_activation_cls(
                        activation=nn.Identity(),
                        input_dim=action_net.out_features,
                        nm_signal_dim=self.mlp_extractor.nm_signal_dim,
                        device=self.device,
                    ),
                ),
                in_keys=["nm_z_actor"],
                # name="action_net_actor",
            ).to(self.device)

        else:
            self.action_net = action_net

        if self.is_value_net_nmd:
            self.value_net = Nmod(
                Seq(
                    value_net,
                    self.nm_activation_cls(
                        activation=nn.Identity(),
                        input_dim=value_net.out_features,
                        nm_signal_dim=self.mlp_extractor.nm_signal_dim,
                        device=self.device,
                    ),
                ),
                in_keys=["nm_z_critic"],
                # name="value_net_critic",
            ).to(self.device)
        else:
            self.value_net = value_net

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        self._build_action_value_networks()

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
