import torch
import torch.nn as nn
from torch.autograd import Function
import gpt_ext


__all__ = [
    'RMSNorm',
]


class rms_norm_cuda_function(Function):
    @staticmethod
    def forward(input, weight):
        return gpt_ext.rms_norm_fw_cuda(input, weight)
    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = gpt_ext.rms_norm_bw_cuda(grad_output, input, weight)
        if not ctx.needs_input_grad[0]:
            grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None
        return grad_input, grad_weight
rms_norm_cuda = rms_norm_cuda_function.apply


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x):
        if x.is_cuda:
            return rms_norm_cuda(x, self.weight)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
