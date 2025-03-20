#include <torch/torch.h>
#include <iostream>
#include "net.h"

double acc(torch::Tensor& preds, torch::Tensor& labels){
  // compute the batch accuracy given the network classification and the labels.
  // TODO
  return 0.;
}

int main() {
  // Set the seed.

  // Create a new Net.
  auto net = std::make_shared<Net>();

  std::cout << "CUDA is available: " << torch::cuda::is_available() << std::endl;
  // TODO: Move your data to the GPU.
  
  // Create a multi-threaded data loader for the MNIST dataset.
  auto data_set = torch::data::datasets::MNIST("../data/")
          .map(torch::data::transforms::Normalize<>(0.5, 0.5))
          .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(data_set),
          torch::data::DataLoaderOptions().batch_size(64).workers(2));

  //TODO: move your network to the GPU

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  for (size_t epoch = 1; epoch <= 20; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      batch.target = batch.target.to(device, false);
      batch.data = batch.data.to(device);
      
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.

      torch::Tensor prediction = net -> forward(batch.data);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      
      torch::Tensor net_choice = prediction.argmax(-1);
      double accuracy = acc(net_choice, batch.target);

      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>()
                  << " | Accuracy:" << accuracy << std::endl;

      }
    }
  }

  (*net).to(torch::kCPU, false);
  // Serialize your model periodically as a checkpoint.
  torch::save(net, "../trained_net/net.pt");
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
      // TODO: Loop over the test data and find the test accuracy
      // The test accuracy is the ratio of currectly identified
      // digits over the total number of digits.
  }
  double test_acc = 0.;
  std::cout << "Total: " << total << " correct: " << correct << std::endl;
  std::cout << "Test accuracy:" << test_acc << std::endl;

}
