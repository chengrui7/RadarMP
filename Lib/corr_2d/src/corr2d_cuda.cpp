#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "corr2d_cuda_kernel.h"

int corr_cuda_forward(
    const at::Tensor& input1,
    const at::Tensor& input2,
    at::Tensor& rbot1,
    at::Tensor& rbot2,
    at::Tensor& output,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply
) {
    // input1, input2: [B, C, H, W]
    TORCH_CHECK(input1.is_cuda() && input2.is_cuda(), "Input must be CUDA tensor");

    int batchSize = input1.size(0);
    int nInputPlane = input1.size(1);
    int height = input1.size(2);
    int width = input1.size(3);

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_;

    int padded_H = height + 2 * pad_size;
    int padded_W = width + 2 * pad_size;

    int out_H = std::ceil((float)(padded_H - 2 * border_size_) / stride1);
    int out_W = std::ceil((float)(padded_W - 2 * border_size_) / stride1);

    int grid_radius = max_displacement / stride2;
    int grid_width = grid_radius * 2 + 1;
    int nOutputPlane = grid_width * grid_width;

    output.resize_({batchSize, nOutputPlane, out_H, out_W}).zero_();
    rbot1.resize_({batchSize, nInputPlane, padded_H, padded_W}).zero_();
    rbot2.resize_({batchSize, nInputPlane, padded_H, padded_W}).zero_();

    auto stream = at::cuda::getCurrentCUDAStream();

    blob_rearrange_ongpu(
        input1.data_ptr<float>(), rbot1.data_ptr<float>(),
        batchSize, nInputPlane, height, width, pad_size, stream);

    blob_rearrange_ongpu(
        input2.data_ptr<float>(), rbot2.data_ptr<float>(),
        batchSize, nInputPlane, height, width, pad_size, stream);

    CorrelateData_ongpu(
        rbot1.data_ptr<float>(),
        rbot2.data_ptr<float>(),
        output.data_ptr<float>(),
        batchSize, out_H, out_W, nOutputPlane,
        max_displacement, grid_radius, grid_width,
        kernel_radius_, kernel_size,
        stride1, stride2,
        padded_H, padded_W,
        nInputPlane,
        corr_type_multiply,
        stream);

    return 1;
}


int corr_cuda_backward(
    const at::Tensor& input1,
    const at::Tensor& input2,
    at::Tensor& rbot1,
    at::Tensor& rbot2,
    const at::Tensor& grad_output,
    at::Tensor& grad_input1,
    at::Tensor& grad_input2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply
) {
    int batchSize = input1.size(0);
    int nInputPlane = input1.size(1);
    int height = input1.size(2);
    int width = input1.size(3);

    long kernel_radius_ = (kernel_size - 1) / 2;
    long border_size_ = max_displacement + kernel_radius_;

    int padded_H = height + 2 * pad_size;
    int padded_W = width + 2 * pad_size;

    int out_H = std::ceil((float)(padded_H - 2 * border_size_) / stride1);
    int out_W = std::ceil((float)(padded_W - 2 * border_size_) / stride1);

    int grid_radius = max_displacement / stride2;
    int grid_width = grid_radius * 2 + 1;
    int nOutputPlane = grid_width * grid_width;

    rbot1.resize_({batchSize, nInputPlane, padded_H, padded_W}).zero_();
    rbot2.resize_({batchSize, nInputPlane, padded_H, padded_W}).zero_();
    grad_input1.resize_({batchSize, nInputPlane, height, width}).zero_();
    grad_input2.resize_({batchSize, nInputPlane, height, width}).zero_();

    auto stream = at::cuda::getCurrentCUDAStream();

    blob_rearrange_ongpu(
        input1.data_ptr<float>(), rbot1.data_ptr<float>(),
        batchSize, nInputPlane, height, width, pad_size, stream);

    blob_rearrange_ongpu(
        input2.data_ptr<float>(), rbot2.data_ptr<float>(),
        batchSize, nInputPlane, height, width, pad_size, stream);

    CorrelateDataBackward_ongpu(
        rbot1.data_ptr<float>(), rbot2.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_input1.data_ptr<float>(), grad_input2.data_ptr<float>(),
        batchSize, out_H, out_W, nOutputPlane,
        max_displacement, grid_radius, grid_width,
        kernel_radius_,
        stride1, stride2,
        height, width,
        padded_H, padded_W,
        nInputPlane, pad_size,
        corr_type_multiply,
        stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("corr_cuda_forward", &corr_cuda_forward, "Correlation 2D forward (CUDA)");
    m.def("corr_cuda_backward", &corr_cuda_backward, "Correlation 2D backward (CUDA)");
}
