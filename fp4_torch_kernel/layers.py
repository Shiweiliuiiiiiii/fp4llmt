import torch
import torch.nn as nn
from fp4_torch_kernel.utils import FP4ToBF16Function

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.empty(out_features, in_features, dtype=torch.bfloat16)
        self.weight = nn.Parameter(weight.view(torch.float4_e2m1fn_x2))
        if bias:
            bias_tensor = torch.empty(out_features, dtype=torch.bfloat16)
            self.bias = nn.Parameter(bias_tensor.view(torch.float4_e2m1fn_x2))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization (here using normal distribution in bfloat16) then cast to float4.
        weight_bf16 = torch.empty_like(self.weight.data.view(torch.bfloat16)).normal_()
        self.weight.data = weight_bf16.view(torch.float4_e2m1fn_x2)
        if self.bias is not None:
            bias_bf16 = torch.empty_like(self.bias.data.view(torch.bfloat16)).normal_()
            self.bias.data = bias_bf16.view(torch.float4_e2m1fn_x2)

    def forward(self, input):
        weight_bf16 = FP4ToBF16Function.apply(self.weight)
        bias_bf16 = FP4ToBF16Function.apply(self.bias)
        return nn.functional.linear(input, weight_bf16, bias_bf16)