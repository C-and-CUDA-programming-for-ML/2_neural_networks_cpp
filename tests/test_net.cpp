//#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include "../source/net.h"

int main() { 
  auto net = std::make_shared<Net>();
  torch::load(net, "../trained_net/net.pt");
  std::cout << "model saved.";
  std::cout << "started tests.";

  auto test_set = torch::data::datasets::MNIST("../data/",
                                               torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.5, 0.5))
          .map(torch::data::transforms::Stack<>());
  // test our network
  auto test_loader = torch::data::make_data_loader(
          test_set,
          torch::data::DataLoaderOptions().batch_size(64).workers(2));

  double correct = 0;
  double total = 0;
  for (auto& test_batch:  *test_loader) {
    torch::Tensor test_batch_data = test_batch.data;
    torch::Tensor test_batch_labels = test_batch.target;
    torch::Tensor test_out = net->forward(test_batch_data);
    torch::Tensor test_out_max = test_out.argmax(-1);
    for (int i = 0; i < test_out_max.size(0); i++){
      total += 1;
      if (test_out_max[i].item<int>() == test_batch_labels[i].item<int>()){
        correct += 1;
      }
    }
  }
  double test_acc = correct / total;
  std::cout << "Total: " << total << " correct: " << correct << std::endl;
  std::cout << "Test accuracy:" << test_acc << std::endl;
  assert(test_acc > 0.9);
  return 0;
}