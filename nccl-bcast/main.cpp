#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <gloo/transport/tcp/device.h>
#include <iostream>
#include <cstring>
#include <memory>
#include <vector>

struct ncclUniqueId {
  char internal[128];
};

void bcast_example(std::shared_ptr<c10d::ProcessGroupGloo> process_group, int rank) {
    ncclUniqueId ncclId;
    if (rank == 0) {
      std::memset(ncclId.internal, 42, sizeof(ncclId.internal));
    }

    // Create a tensor to hold the ncclId data;
    auto ncclId_tensor = torch::from_blob(ncclId.internal,
                                          {static_cast<int64_t>(sizeof(ncclId.internal))}, torch::kByte);

    std::vector<at::Tensor> ncclId_tensor_vector = {ncclId_tensor};
    // Broadcast options
    c10d::BroadcastOptions opts;
    opts.rootRank = 0;

    // Perform the bcast operation
    process_group->broadcast(ncclId_tensor_vector, opts)->wait();

    // Print the result to verify
    if (rank != 0) {
      std::cout << "Rank " << rank << " received ncclId: ";
      for (int i = 0; i < sizeof(ncclId.internal); ++i) {
        std::cout << static_cast<int>(ncclId.internal[i]) << " ";
      }
      std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    // Create a TCPStore instance
    //auto store = std::make_shared<c10d::TCPStore>("127.0.0.1");

    ::gloo::transport::tcp::attr attr;
    attr.iface = "eno1";
    auto gloo_device = ::gloo::transport::tcp::CreateDevice(attr);
    auto options = c10d::ProcessGroupGloo::Options::create();
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

    // Run the bcast example
    bcast_example(process_group, rank);

    return 0;
}
