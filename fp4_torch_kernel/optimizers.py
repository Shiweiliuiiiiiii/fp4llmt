import torch
import math
import torch.nn as nn
from fp4_torch_kernel.utils import FP4ToBF16Function

import torch
import math
import torch.nn as nn
# Assuming FP4ToBF16Function exists and works as intended for gradients
# from fp4_torch_kernel.utils import FP4ToBF16Function

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

        # Check if float4 dtype exists
        self.fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if self.fp4_dtype is None:
             print("Warning: torch.float4_e2m1fn_x2 not found. FP4 specific logic will be disabled.")
             # You might want to raise an error or default to standard Adam behavior entirely

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FP4Adam, self).__init__(params, defaults)

        self.compute_dtype = compute_dtype
        # print(f"FP4Adam: Using {self.compute_dtype} for internal calculations and state.")
        # print(f"FP4Adam: Will handle parameters of type {self.fp4_dtype} with FP4 logic, others with standard logic.")

        # Initialize state (must be stored in compute_dtype)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: # Only optimize parameters that require gradients
                    state = self.state[p]
                    # State initialization: Store step, exp_avg, exp_avg_sq
                    state['step'] = 0
                    # Moment estimates - MUST be stored in compute_dtype, get shape from p
                    if p.dtype == torch.float4_e2m1fn_x2: # Parameters should be converted, otherwise shapes will not match!
                        state['exp_avg'] = torch.zeros_like(p.view(torch.bfloat16), memory_format=torch.preserve_format, dtype=self.compute_dtype)
                        state['exp_avg_sq'] = torch.zeros_like(p.view(torch.bfloat16), memory_format=torch.preserve_format, dtype=self.compute_dtype)
                        state['exp_avg'] = state['exp_avg'].view(torch.float4_e2m1fn_x2)
                        state['exp_avg_sq'] = state['exp_avg_sq'].view(torch.float4_e2m1fn_x2)
                    else:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=self.compute_dtype)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, dtype=self.compute_dtype)

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
                if not p.requires_grad: # Skip params that don't require grad
                    continue

                grad = p.grad.data # Get the gradient data
                original_param_dtype = p.dtype # Store original dtype for casting back

                # --- State Handling (Common) ---
                state = self.state[p]
                # State tensors are already in compute_dtype
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                # --- Prepare Inputs based on Parameter Dtype ---
                is_fp4 = self.fp4_dtype and original_param_dtype == self.fp4_dtype

                if is_fp4:
                    # --- FP4 Parameter Handling ---
                    if grad.dtype != self.fp4_dtype:
                         # This assumes your backward pass correctly generates fp4 grads for fp4 params
                         raise TypeError(f"Gradient dtype mismatch for FP4 parameter! Expected {self.fp4_dtype}, got {grad.dtype}. Check your backward pass (e.g., FP4ToBF16Function.backward).")

                    # Cast FP4 grad and param to compute_dtype using view (bit-reinterpretation)
                    # Ensure shape compatibility for view if compute_dtype is wider
                    expected_compute_elements = p.numel() * p.element_size() // torch.tensor([], dtype=self.compute_dtype).element_size()
                    grad_compute = grad.view(self.compute_dtype)
                    param_compute = p.data.view(self.compute_dtype)
                    exp_avg = exp_avg.view(self.compute_dtype)
                    exp_avg_sq = exp_avg_sq.view(self.compute_dtype)
                    if grad_compute.numel() != expected_compute_elements or param_compute.numel() != expected_compute_elements:
                        raise RuntimeError(f"Shape mismatch after viewing FP4 tensor as {self.compute_dtype}. "
                                           f"Original numel: {p.numel()}, expected compute numel: {expected_compute_elements}, "
                                           f"got grad: {grad_compute.numel()}, param: {param_compute.numel()}. "
                                           f"Ensure tensor size is compatible with view.")


                else:
                    # --- Standard Float Parameter Handling ---
                    # Cast standard grad and param to compute_dtype using .to() (numerical conversion)
                    grad_compute = grad.to(self.compute_dtype)
                    # Create a compute-precision version of the parameter for calculations
                    param_compute = p.data.to(self.compute_dtype)


                # --- Adam Update Logic (Common - operates on compute_dtype tensors) ---

                # Apply weight decay (L2 penalty style) - Use param_compute
                if weight_decay != 0:
                    # Add weight decay in compute_dtype. Note: Adds compute_dtype(param) * weight_decay
                    grad_compute = grad_compute.add(param_compute, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad_compute, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_compute, grad_compute.conj() if grad_compute.is_complex() else grad_compute, value=1 - beta2) # Handle complex grads just in case

                # Denominator calculation
                if exp_avg_sq.is_complex():
                   denom = exp_avg_sq.abs().sqrt()
                else:
                   denom = exp_avg_sq.sqrt()
                denom.div_(math.sqrt(bias_correction2)).add_(eps)


                # Step size
                step_size = lr / bias_correction1

                # Parameter update calculation
                update_value = exp_avg.div(denom)

                # Apply update to parameter (in compute_dtype)
                param_compute.add_(update_value, alpha=-step_size)

                # --- Cast Result Back to Original Dtype ---
                if is_fp4:
                    # Cast back to FP4 using view
                    # Ensure param_compute has the right number of elements to be viewed as original_param_dtype
                    p.data = param_compute.view(original_param_dtype)
                    exp_avg = exp_avg.view(torch.float4_e2m1fn_x2)
                    exp_avg_sq = exp_avg_sq.view(torch.float4_e2m1fn_x2)
                else:
                    # Cast back to original standard float dtype using .to()
                    # Check if inplace update is possible if dtypes match
                    if p.data.dtype == self.compute_dtype:
                         p.data.copy_(param_compute) # More robust than direct assignment if p.data was used directly
                    else:
                         p.data = param_compute.to(original_param_dtype)


        return loss