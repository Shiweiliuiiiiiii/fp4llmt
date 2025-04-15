import torch

class FP4ToBF16Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.view(torch.bfloat16)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.view(torch.float4_e2m1fn_x2)
        return grad_input