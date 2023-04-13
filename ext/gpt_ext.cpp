#include <torch/extension.h>

at::Tensor rms_norm_fw_cuda(const at::Tensor &input, const at::Tensor &weight);
std::tuple<at::Tensor, at::Tensor> rms_norm_bw_cuda(const at::Tensor &grad, const at::Tensor &input, const at::Tensor &weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm_fw_cuda", &rms_norm_fw_cuda, "rms_norm_fw_cuda");
  m.def("rms_norm_bw_cuda", &rms_norm_bw_cuda, "rms_norm_bw_cuda");
}
