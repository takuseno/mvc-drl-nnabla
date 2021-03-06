import numpy as np
import nnabla as nn
import argparse
import gym
import pybullet_envs
import roboschool


from nnabla.ext_utils import get_extension_context
from mvc.envs.wrappers import MuJoCoWrapper
from mvc.controllers.ddpg import DDPGController
from mvc.controllers.eval import EvalController
from mvc.models.metrics import Metrics
from mvc.models.buffer import Buffer
from mvc.noise import OrnsteinUhlenbeckActionNoise
from mvc.view import View
from mvc.interaction import interact

from mvc_nnabla.models.networks.ddpg import DDPGNetwork, DDPGNetworkParams


def main(args):
    if args.gpu is not None:
        ctx = get_extension_context('cudnn', device_id=str(args.gpu))
        nn.set_default_context(ctx)

    # environment
    env = MuJoCoWrapper(gym.make(args.env), args.reward_scale, args.render)
    env.seed(args.seed)
    eval_env = MuJoCoWrapper(gym.make(args.env))
    eval_env.seed(args.seed)
    num_actions = env.action_space.shape[0]

    # network parameters
    params = DDPGNetworkParams(fcs=args.layers, concat_index=args.concat_index,
                               state_shape=env.observation_space.shape,
                               num_actions=num_actions, gamma=args.gamma,
                               tau=args.tau, actor_lr=args.actor_lr,
                               critic_lr=args.critic_lr,
                               batch_size=args.batch_size)

    # deep neural network
    network = DDPGNetwork(params)

    # replay buffer
    buffer = Buffer(args.buffer_size)

    # metrics
    metrics = Metrics(args.name)

    # exploration noise
    noise = OrnsteinUhlenbeckActionNoise(
        np.zeros(num_actions), np.ones(num_actions) * 0.2)

    # controller
    controller = DDPGController(network, buffer, metrics, noise, num_actions,
                                args.batch_size, args.final_steps,
                                args.log_interval, args.save_interval,
                                args.eval_interval)

    # view
    view = View(controller)

    # evaluation
    eval_controller = EvalController(network, metrics, args.eval_episode)
    eval_view = View(eval_controller)

    # save hyperparameters
    metrics.log_parameters(vars(args))

    interact(env, view, eval_env, eval_view)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='training environment')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='reward scaling')
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 64],
                        help='layer units')
    parser.add_argument('--concat-index', type=int, default=1,
                        help='index of layer to concat action at q function')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='soft target update factor')
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                        help='learning rate for actor')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='learning rate for critic')
    parser.add_argument('--buffer-size', type=int, default=10 ** 6,
                        help='size of replay buffer')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--final-steps', type=int, default=10 ** 6,
                        help='the number of training steps')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='log interval')
    parser.add_argument('--save-interval', type=int, default=10 ** 5,
                        help='save interval')
    parser.add_argument('--eval-interval', type=int, default=10 ** 4,
                        help='evaluation interval')
    parser.add_argument('--eval-episode', type=int, default=10,
                        help='the number of evaluation episode')
    parser.add_argument('--name', type=str, default='ddpg',
                        help='name of experiment')
    parser.add_argument('--load', type=str, help='path to model file')
    parser.add_argument('--render', action='store_true',
                        help='show rendered frames')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of environment')
    parser.add_argument('--gpu', type=int, help='GPU device id')
    args = parser.parse_args()
    main(args)
