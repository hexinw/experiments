#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <gloo/transport/tcp/device.h>
#include <iostream>
#include <sstream>
#include <unistd.h>

void all_gather_example(std::shared_ptr<c10d::ProcessGroupGloo> process_group, int rank, int size) {
    char line[2048] = {char((int)'A' + rank), ':', ' ', 'H', 'E', 'L', 'L', 'O'};
    // Create a tensor containing the rank of the current process
    //auto tensor = torch::tensor({rank}, torch::kFloat32); // Keep tensor on CPU
    auto tensor = torch::from_blob((void*)line, {2048}, torch::kUInt8);

    // Prepare a vector to hold the tensors from each process
    std::vector<at::Tensor> input_tensors{tensor};
    //std::cout << "Rank " << rank << " input tensor list size: " << input_tensors.size() << std::endl;

    // Outputs vector (only root process needs to provide storage)
    std::vector<std::vector<torch::Tensor>> output_tensors;
    if (rank == 0) {
      output_tensors.emplace_back();
      for (const auto i : c10::irange(size)) {
        output_tensors.front().emplace_back(at::empty_like(tensor));
      }
    }

    c10d::GatherOptions opts;
    opts.rootRank = 0;

    // Perform the all_gather operation
    process_group->gather(output_tensors, input_tensors, opts)->wait();

    // Print the gathered tensors
    if (rank == 0) {
      std::stringstream ss;
      for (int i = 0; i < size; ++i) {
        //ss << output_tensors[0][i].item().toFloat() << " ";
        ss << output_tensors[0][i].data_ptr<uint8_t>() << " ";
      }
      std::cout << "Rank " << rank << " gathered: " << ss.str() << std::endl;
    }
}

int main(int argc, char** argv) {
    // Create a TCPStore instance
    //auto store = std::make_shared<c10d::TCPStore>("127.0.0.1");

    auto options = c10d::ProcessGroupGloo::Options::create();
    auto gloo_device = ::gloo::transport::tcp::CreateDevice("127.0.0.1");
    options->devices.push_back(gloo_device);

    // Rank and size should be obtained from environment variables for multi-process setup
    int rank = std::atoi(std::getenv("RANK"));
    int size = std::atoi(std::getenv("WORLD_SIZE"));

    // Create an intrusive_ptr from TCPStore
    c10d::TCPStoreOptions store_opts;
    if (rank == 0) {
      store_opts.isServer = true;
    }
    auto store_ptr = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", store_opts);
    //auto store_ptr = c10::make_intrusive<c10d::TCPStore>("127.0.0.1", 29400, 1, true);

    // Create the ProcessGroupGloo
    auto process_group = std::make_shared<c10d::ProcessGroupGloo>(store_ptr, rank, size, options);

    // Run the all_gather example
    all_gather_example(process_group, rank, size);

    return 0;
}
