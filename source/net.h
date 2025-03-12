#include <torch/torch.h>

#ifndef NET_H
#define NET_H

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register three Linear submodules.
    // TODO!
    // call them i.e. fc1, fc2 or fc3.
    // Your last Linear layer should have ten output neurons.
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    torch::Tensor y = torch::zeros_like(x);

    // run the forward pass by accessing your modules
    // and adding activations functions.
    // the last function should be a log_softmax.
    // TODO!
    return y;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif