import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils.parameters as param


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return input_.gt(param.thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input - thresh) < lens
        temp = torch.exp(-(input_ - param.thresh) ** 2 / (2 * param.lens ** 2)) / ((2 * param.lens * 3.141592653589793) ** 0.5)
        return grad_input * temp.float()


class ATan(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of shifted arc-tan function.

        .. math::

                S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{π}\\frac{1}{(1+(πU\\frac{α}{2})^2)}


    α defaults to 2, and can be modified by calling \
        ``surrogate.atan(alpha=2)``.

    Adapted from:

    *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang,
    Y. Tian (2021) Incorporating Learnable Membrane Time Constants
    to Enhance Learning of Spiking Neural Networks. Proc. IEEE/CVF
    Int. Conf. Computer Vision (ICCV), pp. 2661-2671.*"""

    @staticmethod
    def forward(ctx, input_, alpha=2.0):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > param.thresh).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            ctx.alpha
            / 2
            / (1 + (math.pi / 2 * ctx.alpha * input_).pow_(2))
            * grad_input
        )
        return grad, None


class Sigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of sigmoid function.

        .. math::

                S&≈\\frac{1}{1 + {\\rm exp}(-kU)} \\\\
                \\frac{∂S}{∂U}&=\\frac{k
                {\\rm exp}(-kU)}{[{\\rm exp}(-kU)+1]^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.sigmoid(slope=25)``.


    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning
    in Multilayer Spiking
    Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > param.thresh).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * ctx.slope
            * torch.exp(-ctx.slope * input_)
            / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        return grad, None


act_fun = ATan.apply


# membrane potential update

def mem_update(ops, x, mem, spike):
    mem = mem * (1 - spike)  # reset membrane potential if a spike is generated in the previous time stamp.
    mem = mem * param.decay + F.relu(ops(x))
    spike = act_fun(mem)  # act_fun : approximation firing function

    return mem, spike


#  Membrane potential update with pooling included in the operations
def mem_update_pool(ops, pool, x, mem, spike, inorm, norm=False):
    mem = mem * (1 - spike)  # reset membrane potential if a spike is generated in the previous time stamp.
    if norm:
        out = pool(F.relu(inorm(ops(x))))
    else:
        out = pool(F.relu(ops(x)))
    mem = mem * param.decay + out
    spike = act_fun(mem)  # act_fun : approximation firing function

    return mem, spike

def mem_update_reset(ops, x, mem, spike, refractory_period=0, reset_after_spike=False):
    mem *= param.decay
    mem += F.relu(ops(x))
    spike = act_fun(mem)
    mem *= (1 - spike)
    return mem, spike