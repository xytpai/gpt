#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset, 32);
  }
  return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val, T* shared, const int tid) {
  const int lid = tid % 32;
  const int wid = tid / 32;
  val = warp_reduce_sum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < 32) ? shared[lid] : T(0);
  if (wid == 0) {
    val = warp_reduce_sum(val);
  }
  if (tid == 0) {
	shared[0] = val;
  }
  __syncthreads();
  return shared[0];
}

template<typename scalar_t>
__global__ void rms_norm_fw_kernel(
	const scalar_t *input, 
	const float *weight, 
	scalar_t *output, 
	const int hidden_size)
{
	using acc_t = at::acc_type<scalar_t, true>;
	__shared__ acc_t data[32];
	auto offset = blockIdx.x * hidden_size;
	auto input_b = input + offset;
	auto output_b = output + offset;
	int tid = threadIdx.x;
	acc_t sum = 0;
	for (int i = tid; i < hidden_size; i += blockDim.x) {
		acc_t temp = static_cast<acc_t>(input_b[i]);
		sum += temp * temp;
	}
	sum = block_reduce_sum<acc_t>(sum, data, tid);
	auto m = sum/(acc_t)(hidden_size) + 1e-5;
	auto mq = ::sqrt(m);
	for (int i = tid; i < hidden_size; i += blockDim.x) {
		acc_t x = static_cast<acc_t>(input_b[i]);
		acc_t res = x / mq;
		res *= static_cast<acc_t>(weight[i]);
		output_b[i] = static_cast<scalar_t>(res);
	}
}

at::Tensor rms_norm_fw_cuda(const at::Tensor &input, const at::Tensor &weight)
{
	CHECK_CUDA(input);
	CHECK_CUDA(weight);
	TORCH_CHECK(weight.dtype() == at::ScalarType::Float);
	const int batch_size = input.size(0);
	const int t_size = input.size(1);
	const int hidden_size = input.size(2);
	auto output = at::empty_like(input);
	if (output.numel() == 0) {
		return output;
	}
	dim3 grid(batch_size * t_size), block(1024);
	cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream();
	AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "rms_norm_fw_cuda",
      [&]() {
		using acc_t = at::acc_type<scalar_t, true>;
		rms_norm_fw_kernel<<<grid, block, 32 * sizeof(acc_t), cuda_stream>>>(
			input.contiguous().data<scalar_t>(),
			weight.contiguous().data<float>(),
			output.data<scalar_t>(),
			hidden_size);
    });
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return output;
}

template<typename scalar_t>
__global__ void rms_norm_bw_kernel(
	const scalar_t *grad, 
	const scalar_t *input, 
	const float *weight, 
	scalar_t *grad_input,
	scalar_t *grad_weight,
	const int hidden_size)
{
	using acc_t = at::acc_type<scalar_t, true>;
	acc_t eps = 1e-5;
	__shared__ acc_t data[32];
	auto offset = blockIdx.x * hidden_size;
	auto grad_b = grad + offset;
	auto input_b = input + offset;
	auto grad_input_b = grad_input + offset;
	int tid = threadIdx.x;
	acc_t sum = 0;
	for (int i = tid; i < hidden_size; i += blockDim.x) {
		acc_t temp = static_cast<acc_t>(input_b[i]);
		sum += temp * temp;
	}
	sum = block_reduce_sum<acc_t>(sum, data, tid);
	acc_t sum2 = 0;
	for (int i = tid; i < hidden_size; i += blockDim.x) {
		acc_t x = static_cast<acc_t>(input_b[i]);
		acc_t d = static_cast<acc_t>(grad_b[i]);
		sum2 += x * static_cast<acc_t>(weight[i]) * d;
	}
	sum2 = block_reduce_sum<acc_t>(sum2, data, tid);
	acc_t h = (acc_t)(hidden_size);
	acc_t m = sum/h + eps;
	acc_t mq = ::sqrt(m);
	for (int i = tid; i < hidden_size; i += blockDim.x) {
		acc_t x = static_cast<acc_t>(input_b[i]);
		acc_t d = static_cast<acc_t>(grad_b[i]);
		grad_input_b[i] = d * static_cast<double>(weight[i]) * ::pow(m, -0.5) - sum2*x/h*::pow(m,-1.5);
		auto o = x/mq;
		gpuAtomicAdd(grad_weight + i, o*d);
	}
}

std::tuple<at::Tensor, at::Tensor> rms_norm_bw_cuda(const at::Tensor &grad, const at::Tensor &input, const at::Tensor &weight)
{
	TORCH_CHECK(weight.dtype() == at::ScalarType::Float);
	const int batch_size = input.size(0);
	const int t_size = input.size(1);
	const int hidden_size = input.size(2);
	auto grad_input = at::empty_like(input);
	auto grad_weight = at::zeros_like(weight);
	if (input.numel() == 0) {
		return std::forward_as_tuple(grad_input, grad_weight);
	}
	dim3 grid(batch_size * t_size), block(1024);
	cudaStream_t cuda_stream = c10::cuda::getCurrentCUDAStream();
	AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "rms_norm_bw_cuda",
      [&]() {
		using acc_t = at::acc_type<scalar_t, true>;
		rms_norm_bw_kernel<<<grid, block, 32 * sizeof(acc_t), cuda_stream>>>(
			grad.contiguous().data<scalar_t>(),
			input.contiguous().data<scalar_t>(),
			weight.contiguous().data<float>(),
			grad_input.data<scalar_t>(),
			grad_weight.data<scalar_t>(),
			hidden_size);
    });
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return std::forward_as_tuple(grad_input, grad_weight);
}
