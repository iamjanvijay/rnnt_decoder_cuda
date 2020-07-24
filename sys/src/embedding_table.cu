#include<embedding_table.hpp>
#include<logger.hpp>

using namespace s2t::sys;

__global__ void embed_copy(size_t destIndex, size_t srcIndex, float_t* src, float_t* dest)
{
	// int i = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("Hello from block %d, thread %d %d\n", blockIdx.x, threadIdx.x, blockDim.x);

	// TBD:: put a check on length

	size_t gpu_srcIndex = srcIndex*blockDim.x + threadIdx.x;
	size_t gpu_destIndex = destIndex*blockDim.x + threadIdx.x;

	dest[gpu_destIndex] = src[gpu_srcIndex];
}

void embedding_table::lookup(cudnnHandle_t& cudnn, const std::vector<size_t>& seq, gpu_float_array& output)
{
	output.reshape(seq.size(), d_table.shape[1]);

	// can we make a single kernel call ?
	for(size_t index = 0;index < seq.size();++index)
	{
		embed_copy<<<1, output.shape[1]>>>(index, seq[index], d_table.ptr, output.ptr);
	}
}