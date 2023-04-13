import torch
import torch.nn as nn
import gpt_ext
import layers


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def test_rms_norm_fw_cuda(dtype):
    print('test_rms_norm_fw_cuda', dtype)
    modelb = RMSNorm(64)
    model0 = RMSNorm(768)
    model1 = RMSNorm(2048)
    model2 = RMSNorm(4096)
    model3 = RMSNorm(16384)
    model4 = RMSNorm(16384*4+123)
    hszs = [64, 768, 2048, 4096, 16384, 16384*4+123]
    ls = [modelb, model0, model1, model2, model3, model4]
    for i, model in enumerate(ls):
        hsz = hszs[i]
        input_case = torch.rand(4, 1024, hsz).to(dtype)
        output_ref = model(input_case)
        input_case = input_case.cuda()
        w = model.weight.cuda()
        output = gpt_ext.rms_norm_fw_cuda(input_case, w)
        diff = (output.cpu() - output_ref).abs().max().item()
        print('diff', diff)
        assert diff < 1e-2


def test_rms_norm_bw_cuda(dtype):
    print('test_rms_norm_bw_cuda', dtype)
    modelb = RMSNorm(64)
    model0 = RMSNorm(768)
    model1 = RMSNorm(2048)
    model2 = RMSNorm(4096)
    model3 = RMSNorm(16384)
    model4 = RMSNorm(16384*4+123)
    hszs = [64, 768, 2048, 4096, 16384, 16384*4+123]
    ls = [modelb, model0, model1, model2, model3, model4]
    for i, model in enumerate(ls):
        hsz = hszs[i]
        model_ = layers.RMSNorm(hsz).cuda()
        # model_.weight[:] = model.weight

        input_case = torch.randn(4, 1024, hsz)
        input_case_ = input_case.clone().detach().cuda()
        input_case.requires_grad = True
        input_case_.requires_grad = True

        k = torch.randn_like(input_case)
        
        loss = model(input_case)
        loss.backward(k)
        grad_input_ref = input_case.grad
        grad_w_ref = model.weight.grad
        
        loss_ = model_(input_case_)
        loss_.backward(k.cuda())
        grad_input = input_case_.grad
        grad_w = model_.weight.grad

        diff0 = (grad_input.cpu() - grad_input_ref).abs().max().item()
        diff1 = (grad_w.cpu() - grad_w_ref).abs().max().item()

        print(diff0, diff1)
        assert diff0 < 1e-3
        assert diff1 < 1e-3


if __name__ == '__main__':
    test_rms_norm_fw_cuda(torch.float)
    test_rms_norm_fw_cuda(torch.half)
    test_rms_norm_fw_cuda(torch.bfloat16)
    test_rms_norm_bw_cuda(torch.float)
    test_rms_norm_bw_cuda(torch.half)
    print('ok')
