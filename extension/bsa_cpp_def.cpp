#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor query_blocks);