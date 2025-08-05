#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "corr2d_cuda_kernel.h"

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
                                    i < (n); \
                                    i += blockDim.x * gridDim.x)

// 2D Forward Correlation Kernel
__global__ void CorrelateDataKernel2D(
    const int count,
    const int batch_size,
    const int out_H, const int out_W,
    const int nOutputChannels,
    const int max_displacement,
    const int grid_radius,
    const int grid_width,
    const int kernel_radius,
    const int kernel_size,
    const int stride1,
    const int stride2,
    const int padded_H,
    const int padded_W,
    const int nInputChannels,
    const float* __restrict__ rbot1,
    const float* __restrict__ rbot2,
    float* __restrict__ output
) {
    CUDA_KERNEL_LOOP(index, count) {
        const int w = index % out_W;
        const int h = (index / out_W) % out_H;
        const int c = (index / (out_W * out_H)) % nOutputChannels;
        const int b = index / (nOutputChannels * out_W * out_H);

        const int ph = h * stride1 + max_displacement + kernel_radius;
        const int pw = w * stride1 + max_displacement + kernel_radius;

        const int offset_h = ((c / grid_width) - grid_radius) * stride2;
        const int offset_w = (c % grid_width - grid_radius) * stride2;

        float sum = 0.0f;

        for (int ch = 0; ch < nInputChannels; ++ch) {
            for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
                for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                    int h1 = ph + ky;
                    int w1 = pw + kx;

                    int h2 = h1 + offset_h;
                    int w2 = w1 + offset_w;

                    if (h2 >= 0 && h2 < padded_H &&
                        w2 >= 0 && w2 < padded_W) {
                        int idx1 = ((b * nInputChannels + ch) * padded_H + h1) * padded_W + w1;
                        int idx2 = ((b * nInputChannels + ch) * padded_H + h2) * padded_W + w2;
                        sum += rbot1[idx1] * rbot2[idx2];
                    }
                }
            }
        }

        const int out_idx = ((b * nOutputChannels + c) * out_H + h) * out_W + w;
        output[out_idx] = sum;
    }
}

// Launch wrapper for 2D forward
void CorrelateData_ongpu(
    const float* rbot1,
    const float* rbot2,
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
    cudaStream_t stream
) {
    if (corr_type_multiply != 1) {
        printf("Only multiply correlation is supported in 2D version.\n");
        return;
    }

    const int count = batch_size * nOutputChannels * out_H * out_W;
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    CorrelateDataKernel2D<<<blocks, threads, 0, stream>>>(
        count,
        batch_size,
        out_H, out_W,
        nOutputChannels,
        max_displacement,
        grid_radius,
        grid_width,
        kernel_radius,
        kernel_size,
        stride1, stride2,
        padded_H, padded_W,
        nInputChannels,
        rbot1, rbot2, output
    );
}

__global__ void BlobRearrange2D(
    const int nthreads,
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int channels,
    const int H,
    const int W,
    const int padded_H,
    const int padded_W,
    const int pad
) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int w = index % W;
        int h = (index / W) % H;
        int c = (index / (W * H)) % channels;
        int b = index / (channels * H * W);

        int input_idx = ((b * channels + c) * H + h) * W + w;
        int output_idx = ((b * channels + c) * padded_H + (h + pad)) * padded_W + (w + pad);
        output[output_idx] = input[input_idx];
    }
}

void blob_rearrange_ongpu(
    const float* input,
    float* output,
    const int batch,
    const int channels,
    const int H,
    const int W,
    const int pad,
    cudaStream_t stream
) {
    const int count = batch * channels * H * W;
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    int padded_H = H + 2 * pad;
    int padded_W = W + 2 * pad;

    BlobRearrange2D<<<blocks, threads, 0, stream>>>(
        count, input, output,
        batch, channels,
        H, W,
        padded_H, padded_W, pad
    );
}


__global__ void CorrelateDataBackwardKernel2D(
    const int count,
    const int batch_size,
    const int out_H, const int out_W,
    const int nOutputChannels,
    const int max_displacement,
    const int grid_radius,
    const int grid_width,
    const int kernel_radius,
    const int stride1,
    const int stride2,
    const int H, const int W,
    const int padded_H, const int padded_W,
    const int channels,
    const int pad,
    const float* __restrict__ rbot1,
    const float* __restrict__ rbot2,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input1,
    float* __restrict__ grad_input2
) {
    CUDA_KERNEL_LOOP(index, count) {
        const int w = index % W;
        const int h = (index / W) % H;
        const int c = (index / (W * H)) % channels;
        const int b = index / (channels * H * W);

        float sum1 = 0.0f;
        float sum2 = 0.0f;

        const int ph = h + pad;
        const int pw = w + pad;

        for (int o_h = -grid_radius; o_h <= grid_radius; ++o_h) {
            for (int o_w = -grid_radius; o_w <= grid_radius; ++o_w) {
                int h2 = ph + o_h * stride2;
                int w2 = pw + o_w * stride2;

                if (h2 >= kernel_radius && h2 < padded_H - kernel_radius &&
                    w2 >= kernel_radius && w2 < padded_W - kernel_radius) {

                    int tc = (o_h + grid_radius) * grid_width + (o_w + grid_radius);
                    int out_h = (ph - kernel_radius - max_displacement) / stride1;
                    int out_w = (pw - kernel_radius - max_displacement) / stride1;

                    if (out_h >= 0 && out_h < out_H && out_w >= 0 && out_w < out_W) {
                        int g_idx = ((b * nOutputChannels + tc) * out_H + out_h) * out_W + out_w;

                        int idx1 = ((b * channels + c) * padded_H + ph) * padded_W + pw;
                        int idx2 = ((b * channels + c) * padded_H + h2) * padded_W + w2;

                        float g = grad_output[g_idx];
                        sum1 += g * rbot2[idx2];
                        sum2 += g * rbot1[idx1];
                    }
                }
            }
        }

        int grad_idx = ((b * channels + c) * H + h) * W + w;
        grad_input1[grad_idx] = sum1;
        grad_input2[grad_idx] = sum2;
    }
}

void CorrelateDataBackward_ongpu(
    const float* rbot1,
    const float* rbot2,
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
    cudaStream_t stream
) {
    if (corr_type_multiply != 1) {
        printf("Only multiply correlation supported.\n");
        return;
    }

    int count = batch_size * channels * H * W;
    int threads = 512;
    int blocks = (count + threads - 1) / threads;

    CorrelateDataBackwardKernel2D<<<blocks, threads, 0, stream>>>(
        count, batch_size,
        out_H, out_W,
        nOutputChannels,
        max_displacement,
        grid_radius,
        grid_width,
        kernel_radius,
        stride1, stride2,
        H, W,
        padded_H, padded_W,
        channels, pad,
        rbot1, rbot2, grad_output,
        grad_input1, grad_input2
    );
}