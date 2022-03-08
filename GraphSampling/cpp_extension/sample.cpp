#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>
#include <stdlib.h> 
#include <ATen/NativeFunctions.h>
#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/core/Generator.h>
#include <ATen/CPUGeneratorImpl.h>


torch::Tensor edge_sample2(torch::Tensor p_cumsum, int64_t batch_size, c10::optional<at::Generator> generator){
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const float* p_cumsum_ptr = p_cumsum.data_ptr<float>();
    torch::Tensor sampled_indices = torch::empty({batch_size, }, options);
    auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(generator, at::detail::getDefaultCPUGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    for (int64_t i = 0; i < batch_size; i++) {
      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);
      int left_pointer = 0;
      int right_pointer = p_cumsum.numel();
      int mid_pointer;
      int sample_idx;
      float cum_prob;
      while(right_pointer - left_pointer > 0) {
        mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        cum_prob = p_cumsum_ptr[mid_pointer];
        if (cum_prob < uniform_sample) {
          left_pointer = mid_pointer + 1;}
        else {right_pointer = mid_pointer;}
      }
      sample_idx = left_pointer;
      sampled_indices[i] = sample_idx;
    }
    return sampled_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_sample2", &edge_sample2, "fast edge sampler");
  // m.def("edge_sample", &edge_sample, "original edge sampler");
}