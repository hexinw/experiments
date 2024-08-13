#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <gloo/transport/tcp/device.h>
#include <iostream>
#include <sstream>
#include <unistd.h>

template<typename T>                                                                                            
struct torch_type;                                                                                              
                                                                                                                
template<>                                                                                                      
struct torch_type<long long> {                                                                                  
  static at::ScalarType type() { return at::kLong; }                                                            
  static long long value(const at::Tensor& tensor) { return tensor.item().toLong(); }                           };                                                                                                              
                                                                                                                
template<>                                                                                                      
struct torch_type<double> {                                                                                     
  static at::ScalarType type() { return at::kDouble; }                                                          
  static double value(const at::Tensor& tensor) { return tensor.item().toDouble(); }                            
};                                                                                                              
                                                                                                                
template<typename T>                                                                                            
at::Tensor create_tensor_from_blob(T* data, int64_t size) {                                                     
  return torch::from_blob(data, {size}, torch_type<T>::type());                                                 
}                                                                                                               

void all_reduce_example(std::shared_ptr<c10d::ProcessGroupGloo> process_group, int rank, int size) {
    static long long accumulator = rank + 1000000;

    // Create a tensor containing the rank of the current process
    //auto tensor = torch::from_blob(&accumulator, {1}, torch::kLong);
    //auto tensor = torch::tensor({rank + 1000000}, torch::kLong); // Keep tensor on CPU
    auto tensor = create_tensor_from_blob(&accumulator, 1);

    // Prepare a vector to hold the tensors from each process
    std::vector<at::Tensor> input_tensors{tensor};
    //std::cout << "Rank " << rank << " input tensor list size: " << input_tensors.size() << std::endl;

    // Create AllreduceOptions
    c10d::AllreduceOptions opts;
    opts.reduceOp = c10d::ReduceOp::SUM;

    // Perform the all_reduce operation
    process_group->allreduce(input_tensors, opts)->wait();

    //std::cout << "Reduced value: " << input_tensors[0].item().toLong() << std::endl;
    std::cout << "Reduced value: " << accumulator << std::endl;
    //std::cout << "Reduced value: " << torch_type<long long>::value(input_tensors[0]) << std::endl;
    
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

    // Run the all_reduce example
    all_reduce_example(process_group, rank, size);

    return 0;
}
