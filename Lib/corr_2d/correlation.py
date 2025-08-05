import torch
import torch.nn as nn
from torch.autograd import Function
import corr2d_cuda 

class CorrelationFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, 
                max_displacement, stride1, stride2, corr_multiply):

        if input1.dim() != 4 or input2.dim() != 4:
            raise ValueError("Input tensors must be 4D (B,C,H,W)")
        if input1.shape != input2.shape:
            raise ValueError("Input shapes must match")
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        rbot1 = torch.empty(0, device=input1.device, dtype=input1.dtype)
        rbot2 = torch.empty(0, device=input1.device, dtype=input1.dtype)
        output = torch.empty(0, device=input1.device, dtype=input1.dtype)

        corr2d_cuda.corr_cuda_forward(
            input1, input2, rbot1, rbot2, output,
            pad_size, kernel_size, max_displacement,
            stride1, stride2, corr_multiply)

        ctx._rbot1 = rbot1  # optional if used in backward
        ctx._rbot2 = rbot2

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        rbot1 = torch.empty(0, device=input1.device, dtype=input1.dtype)
        rbot2 = torch.empty(0, device=input1.device, dtype=input1.dtype)

        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        corr2d_cuda.corr_cuda_backward(
            input1, input2, rbot1, rbot2, grad_output,
            grad_input1, grad_input2,
            ctx.pad_size, ctx.kernel_size,
            ctx.max_displacement, ctx.stride1,
            ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None

class Correlation2D(nn.Module):
    def __init__(self, pad_size=3, kernel_size=3,
                 max_displacement=1, stride1=1, stride2=1, corr_multiply=1):
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return CorrelationFunction.apply(
            input1, input2,
            self.pad_size, self.kernel_size,
            self.max_displacement, self.stride1,
            self.stride2, self.corr_multiply
        )