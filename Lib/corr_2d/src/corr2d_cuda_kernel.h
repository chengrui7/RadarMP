#ifndef CORR2D_CUDA_KERNEL_H
#define CORR2D_CUDA_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Forward pass for 2D correlation
void CorrelateData_ongpu(
    const float* input1,
    const float* input2,
    float* output,
    int batch_size,
    int out_H, int out_W,
    int nOutputChannels,
    int max_displacement,
    int grid_radius,
    int grid_width,
    int kernel_radius,
    int kernel_size,
    int stride1,
    int stride2,
    int padded_H,
    int padded_W,
    int nInputChannels,
    int corr_type_multiply,
    cudaStream_t stream);

// Backward pass for 2D correlation
void CorrelateDataBackward_ongpu(
    const float* input1,
    const float* input2,
    const float* grad_output,
    float* grad_input1,
    float* grad_input2,
    int batch_size,
    int out_H, int out_W,
    int nOutputChannels,
    int max_displacement,
    int grid_radius,
    int grid_width,
    int kernel_radius,
    int stride1,
    int stride2,
    int H, int W,
    int padded_H, int padded_W,
    int channels,
    int pad,
    int corr_type_multiply,
    cudaStream_t stream);

// Data rearrangement (padding) for 2D inputs
void blob_rearrange_ongpu(
    const float* input,
    float* output,
    int batch,
    int channels,
    int H,
    int W,
    int pad,
    cudaStream_t stream);

#endif // CORR2D_CUDA_KERNEL_H