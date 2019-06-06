from collections import namedtuple

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import numpy as np

from nnabla.initializer import BaseInitializer, UniformInitializer
from mvc.action_output import ActionOutput
from mvc.models.networks.base_network import BaseNetwork
from mvc.misc.assertion import assert_scalar
from mvc_nnabla.parametric_function import deterministic_policy_function
from mvc_nnabla.parametric_function import q_function


class Initializer(BaseInitializer):
    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __call__(self, shape):
        fan_in = int(shape[0])
        val = 1 / np.sqrt(fan_in)
        return np.random.uniform(-val, val, shape)


def build_critic_loss(q_t, rewards_tp1, q_tp1, dones_tp1, gamma):
    assert_scalar(q_t)
    assert_scalar(rewards_tp1)
    assert_scalar(q_tp1)
    assert_scalar(dones_tp1)

    target = rewards_tp1 + gamma * q_tp1 * (1.0 - dones_tp1)
    loss = F.mean((target - q_t) ** 2)
    return loss


def build_target_update(src, dst, tau):
    exprs = []
    for src_var, dst_var in zip(src.values(), dst.values()):
        exprs.append(F.assign(dst_var, dst_var * (1.0 - tau) + src_var * tau))
    return F.sink(*exprs)


def build_optim(loss, learning_rate, params):
    optimizer = S.Adam(learning_rate, eps=1e-8)
    optimizer.set_parameters(params)
    return optimizer


DDPGNetworkParams = namedtuple(
    'DDPGNetworkParams', ('fcs', 'concat_index', 'state_shape', 'num_actions',
                          'gamma', 'tau', 'actor_lr', 'critic_lr',
                          'batch_size'))


class DDPGNetwork(BaseNetwork):
    def __init__(self, params):
        self.params = params
        self._build(params)

    def _infer(self, **kwargs):
        self.infer_obs_t.d = np.array([kwargs['obs_t']])
        self.infer_sink.forward(clear_buffer=True)
        action = self.infer_policy_t.d.copy()[0]
        value = self.infer_q_t.d.copy()[0][0]
        return ActionOutput(action=action, log_prob=None, value=value)

    def _update(self, **kwargs):
        # critic update
        self.obs_t.d = kwargs['obs_t']
        self.actions_t.d = kwargs['actions_t']
        self.rewards_tp1.d = kwargs['rewards_tp1']
        self.obs_tp1.d = kwargs['obs_tp1']
        self.dones_tp1.d = kwargs['dones_tp1']
        self.critic_loss.forward()
        self.critic_solver.zero_grad()
        self.critic_loss.backward(clear_buffer=True)
        self.critic_solver.update()
        critic_loss = self.critic_loss.d.copy()

        # actor update
        self.obs_t.d = kwargs['obs_t']
        self.actor_loss.forward()
        self.actor_solver.zero_grad()
        self.actor_loss.backward(clear_buffer=True)
        self.actor_solver.update()
        actor_loss = self.actor_loss.d.copy()

        # target update
        self.update_target.forward(clear_buffer=True)

        return critic_loss, actor_loss

    def _build(self, params):
        with nn.parameter_scope('ddpg'):
            last_initializer = UniformInitializer((-3e-3, 3e-3))

            # inference
            self.infer_obs_t = nn.Variable((1,) + params.state_shape)
            infer_raw_policy_t = deterministic_policy_function(
                params.fcs, self.infer_obs_t, params.num_actions, F.tanh,
                w_init=Initializer(), last_w_init=last_initializer,
                last_b_init=last_initializer, scope='actor')
            self.infer_policy_t = F.tanh(infer_raw_policy_t)

            self.infer_q_t = q_function(
                params.fcs, self.infer_obs_t, self.infer_policy_t,
                params.concat_index, F.tanh, w_init=Initializer(),
                last_w_init=last_initializer, last_b_init=last_initializer,
                scope='critic')

            self.infer_sink = F.sink(self.infer_policy_t, self.infer_q_t)

            # training
            self.obs_t = nn.Variable((params.batch_size,) + params.state_shape)
            self.actions_t = nn.Variable((params.batch_size, params.num_actions))
            self.rewards_tp1 = nn.Variable((params.batch_size,))
            self.obs_tp1 = nn.Variable((params.batch_size,) + params.state_shape)
            self.dones_tp1 = nn.Variable((params.batch_size,))

            raw_policy_t = deterministic_policy_function(
                params.fcs, self.obs_t, params.num_actions, F.tanh,
                w_init=Initializer(), last_w_init=last_initializer,
                last_b_init=last_initializer, scope='actor')
            policy_t = F.tanh(raw_policy_t)
            raw_policy_tp1 = deterministic_policy_function(
                params.fcs, self.obs_tp1, params.num_actions, F.tanh,
                w_init=Initializer(), last_w_init=last_initializer,
                last_b_init=last_initializer, scope='target_actor')
            policy_tp1 = F.tanh(raw_policy_tp1)

            q_t = q_function(
                params.fcs, self.obs_t, self.actions_t,
                params.concat_index, F.tanh, w_init=Initializer(),
                last_w_init=last_initializer, last_b_init=last_initializer,
                scope='critic')
            q_t_with_actor = q_function(
                params.fcs, self.obs_t, policy_t, params.concat_index,
                F.tanh, w_init=Initializer(), last_w_init=last_initializer,
                last_b_init=last_initializer, scope='critic')
            q_tp1 = q_function(
                params.fcs, self.obs_tp1, policy_tp1, params.concat_index,
                F.tanh, w_init=Initializer(), last_w_init=last_initializer,
                last_b_init=last_initializer, scope='target_critic')

            # prepare for loss calculation
            rewards_tp1 = F.reshape(self.rewards_tp1, (-1, 1))
            dones_tp1 = F.reshape(self.dones_tp1, (-1, 1))

            # critic loss
            self.critic_loss = build_critic_loss(q_t, rewards_tp1, q_tp1,
                                                 dones_tp1, params.gamma)
            # actor loss
            self.actor_loss = -F.mean(q_t_with_actor)

            # parameters
            with nn.parameter_scope('actor'):
                actor_params = nn.get_parameters()
            with nn.parameter_scope('target_actor'):
                target_actor_params = nn.get_parameters()
            with nn.parameter_scope('critic'):
                critic_params = nn.get_parameters()
            with nn.parameter_scope('target_critic'):
                target_critic_params = nn.get_parameters()

            # optimization
            self.critic_solver = build_optim(
                self.critic_loss, params.critic_lr, critic_params)
            self.actor_solver = build_optim(
                self.actor_loss, params.actor_lr, actor_params)

            # target update
            update_actor_target = build_target_update(
                actor_params, target_actor_params, params.tau)
            update_critic_target = build_target_update(
                critic_params, target_critic_params, params.tau)
            self.update_target = F.sink(
                update_actor_target, update_critic_target)


    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return [
            'obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1'
        ]
