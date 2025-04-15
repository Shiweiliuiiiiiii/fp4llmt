import torch
import torch.nn as nn
from fp4_torch_kernel.utils import FP4ToBF16Function

class FP4Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, compute_dtype=torch.bfloat16):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FP4Adam, self).__init__(params, defaults)

        self.compute_dtype = compute_dtype # Dtype for calculations and state
        print(f"FP4Adam: Using {self.compute_dtype} for internal calculations and state.")

        # Initialize state (must be stored in compute_dtype)
        for group in self.param_groups:
            for p in group['params']:
                if p.dtype != torch.float4_e2m1fn_x2:
                     print(f"Warning: Parameter found with dtype {p.dtype}, expected float4. Optimizer might misbehave.")
                state = self.state[p]
                # State initialization: Store step, exp_avg, exp_avg_sq
                state['step'] = 0
                # Moment estimates - MUST be stored in higher precision
                state['exp_avg'] = torch.zeros_like(p.view(self.compute_dtype), memory_format=torch.preserve_format, dtype=self.compute_dtype)
                state['exp_avg_sq'] = torch.zeros_like(p.view(self.compute_dtype), memory_format=torch.preserve_format, dtype=self.compute_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # --- Cast Inputs to Compute Precision ---
                # Parameter is float4, Gradient is float4
                if p.grad.dtype != torch.float4_e2m1fn_x2:
                     raise TypeError(f"Gradient dtype mismatch! Expected float4, got {p.grad.dtype}. Ensure Float4ToBF16Function.backward produces float4.")

                # Cast grad and param to compute_dtype for calculations
                grad_compute = p.grad.data.view(self.compute_dtype)
                param_compute = p.data.view(self.compute_dtype)

                # --- State Handling ---
                state = self.state[p]
                # State tensors are already in compute_dtype
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # --- Adam Update Logic (in compute_dtype) ---

                # Apply weight decay (L2 penalty style)
                if weight_decay != 0:
                    grad_compute = grad_compute.add(param_compute, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad_compute, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_compute, grad_compute, value=1 - beta2)

                # Denominator calculation
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Step size
                step_size = lr / bias_correction1

                # Parameter update calculation
                update_value = exp_avg.div(denom)

                # Apply update to parameter (still in compute_dtype)
                param_compute.add_(update_value, alpha=-step_size)

                # --- Cast Result Back to Float4 ---
                p.data = param_compute.view(p.dtype) # Write back updated value in float4

        return loss