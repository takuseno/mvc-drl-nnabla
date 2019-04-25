import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from nnabla.initializer import BaseInitializer, ConstantInitializer
from nnabla.parameter import get_parameter_or_create


class OrthogonalInitializer(BaseInitializer):

    r"""Generates an orthogonal matrix weights proposed by Saxe et al.
    Args:
        gain (float): scaling factor which should be decided depending on a type of units.
        rng (numpy.random.RandomState): Random number generator.
    Example:
    .. code-block:: python
        import numpy as np
        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I
        x = nn.Variable([60,1,28,28])
        w = I.OrthogonalInitializer(np.sqrt(2.0))
        b = I.ConstantInitializer(0.0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')
    References:
        * `Saxe, et al. Exact solutions to the nonlinear dynamics of
          learning in deep linear neural networks.
          <https://arxiv.org/abs/1312.6120>`_
    """

    def __init__(self, gain=1.0, rng=None):
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.gain = gain

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.gain)

    def __call__(self, shape):
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        x = self.rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(x, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape).astype('float32') * self.gain


def _make_fcs(fcs, inpt, activation, w_init=None):
    if w_init is None:
        w_init = OrthogonalInitializer(np.sqrt(2.0))
    out = inpt
    with nn.parameter_scope('hiddens'):
        for i, hidden in enumerate(fcs):
            out = PF.affine(out, hidden, w_init=w_init,
                            name='hidden{}'.format(i))
            out = activation(out)
    return out


#def stochastic_policy_function(fcs,
#                               inpt,
#                               num_actions,
#                               activation=F.tanh,
#                               share=False,
#                               w_init=None,
#                               last_w_init=None,
#                               last_b_init=None,
#                               scope='policy'):
#    with nn.parameter_scope(scope):
#        out = _make_fcs(fcs, inpt, activation, w_init)
#        mean = PF.affine(out, num_actions, activation=None, w_init=last_w_init,
#                         b_init=last_b_init, name='mean')
#
#        if share:
#            logstd = PF.affine(out, num_actions, activation=None,
#                               w_init=last_w_init, b_init=last_b_init,
#                               name='logstd')
#            clipped_logstd = F.clip_by_value(
#                logstd, F.constant(-20, logstd.shape),
#                F.constant(2, logstd.shape))
#            std = F.exp(clipped_logstd)
#        else:
#            logstd = get_parameter_or_create(
#                'logstd', shape=[1, num_actions],
#                initializer=ConstantInitializer(0.0))
#            std = F.constant(0.0, mean.shape) + F.exp(logstd)
#
#        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
#    return dist


def deterministic_policy_function(fcs,
                                  inpt,
                                  num_actions,
                                  activation=F.tanh,
                                  w_init=None,
                                  last_w_init=None,
                                  last_b_init=None,
                                  scope='policy'):
    if last_b_init is None:
        last_b_init = ConstantInitializer(0.0)

    with nn.parameter_scope(scope):
        out = _make_fcs(fcs, inpt, activation, w_init)
        policy = PF.affine(out, num_actions, w_init=last_w_init,
                           b_init=last_b_init, name='output')
    return policy


def value_function(fcs,
                   inpt,
                   activation=F.tanh,
                   w_init=None,
                   last_w_init=None,
                   last_b_init=None,
                   scope='value'):
    with nn.parameter_scope(scope):
        out = _make_fcs(fcs, inpt, activation, w_init)
        value = PF.affine(out, 1, w_init=last_w_init, b_init=last_b_init,
                          name='output')
    return value


def q_function(fcs,
               inpt,
               action,
               concat_index,
               activation=F.tanh,
               w_init=None,
               last_w_init=None,
               last_b_init=None,
               scope='action_value'):
    if last_b_init is None:
        last_b_init = ConstantInitializer(0.0)

    with nn.parameter_scope(scope):
        out = inpt
        with nn.parameter_scope('hiddens'):
            for i, hidden in enumerate(fcs):
                if i == concat_index:
                    out = F.concatenate(out, action, axis=1)
                out = PF.affine(out, hidden, w_init=w_init,
                                name='hidden{}'.format(i))
                out = activation(out)
        value = PF.affine(out, 1, w_init=last_w_init, b_init=last_b_init,
                          name='output')
    return value
