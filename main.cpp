#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor forward_decode(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", torch::wrap_pybind_function(forward), "forward");
  m.def("forward_decode", torch::wrap_pybind_function(forward_decode),
        "forward_decode");
}