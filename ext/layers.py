import torch 
import torch.nn as nn
from torch.autograd import Function
import gpt_ext


class rms_norm_function(Function):
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
rms_norm_func = rms_norm_function.apply


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm_func(x, self.weight)
