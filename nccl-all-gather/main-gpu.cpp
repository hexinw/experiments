#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <iostream>
#include <unistd.h>

void all_gather_example(std::shared_ptr<torch::distributed::c10d::ProcessGroup> process_group, int rank, int size) {
    // Create a tensor containing the rank of the current process
    auto tensor = torch::tensor({rank}, torch::kFloat32).cuda(); // Move tensor to GPU

    // Prepare a vector to hold the gathered tensors
    std::vector<torch::Tensor> gathered_tensors(size);

    // Perform the all_gather operation
    process_group->allgather(gathered_tensors, tensor).get();

    // Print the gathered tensors
    for (const auto& t : gathered_tensors) {
        std::cout << "Rank " << rank << " gathered: " << t.cpu() << std::endl; // Move tensor back to CPU for printing
    }
}

int main(int argc, char** argv) {
    // Initialize the process group
    torch::distributed::c10d::Store store("127.0.0.1:29500", 4);
    auto options = torch::distributed::c10d::ProcessGroupNCCL::Options::create();

    // Rank and size should be obtained from environment variables for multi-process setup
    int rank = std::atoi(std::getenv("RANK"));
    int size = std::atoi(std::getenv("WORLD_SIZE"));

    // Create the ProcessGroupNCCL
    auto process_group = std::make_shared<torch::distributed::c10d::ProcessGroupNCCL>(store, rank, size, options);

    // Run the all_gather example
    all_gather_example(process_group, rank, size);

    return 0;
}
