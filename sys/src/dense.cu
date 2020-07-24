#include<dense.hpp>

using namespace s2t::sys;

__global__ void dense_add(size_t sz, float_t* src, float_t* dest, size_t dst_offset)
{
	size_t srcIndex = threadIdx.x;
	size_t destIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(destIndex < sz)
	{
		dest[destIndex + dst_offset] += src[srcIndex];
	}
}

__global__ void dense_add_conv(size_t sz, float_t* src, float_t* dest, size_t audio_len)
{
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < sz)
	{
		dest[index] += src[index/audio_len];
	}
}


dense::dense() { }

void dense::init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias)
{
	checkCUBLAS(cublasCreate (& handle ));
    (cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

	// load kernel
	d_kernel.init(h_kernel.shape);
	cudaMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
	
	// load bias
	d_bias.init(h_bias.shape);
	cudaMemcpy(d_bias.ptr, h_bias.data<float_t>(), d_bias.size()*sizeof(float_t), cudaMemcpyHostToDevice);

	hasbias = true; 
}

void dense::init(const cnpy::NpyArray& h_kernel)
{
	checkCUBLAS(cublasCreate (& handle ));
    (cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

	// load kernel
	d_kernel.init(h_kernel.shape);
	cudaMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), cudaMemcpyHostToDevice);

	hasbias = false;
}

//  This computes - d_input(n*k) * d_kernel (k*m) = d_output (n*m)

void dense::operator () (cudnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, size_t ptr_offset)
{
	const float alpha = 1, beta = 0;
	size_t m = d_kernel.shape[1]; 
	size_t k = d_kernel.shape[0];
	size_t n = d_input.shape[0];

	//std::cout<<m<<":"<<k<<":"<<":"<<n<<std::endl;

	if(n == 1 || d_input.shape[1] == 1)
	{
		checkCUBLAS(cublasSgemv(handle,
			CUBLAS_OP_N,
			m,
			k,
			&alpha,
			d_kernel.ptr,
			m,
			d_input.ptr,
			1,
			&beta,
			d_output.ptr + ptr_offset,
			1));

	// add bias	
		if(hasbias)
		{
			dense_add<<<1, m>>>(m, d_bias.ptr, d_output.ptr, ptr_offset);
		}
	}
	else
	{
		// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_kernel.ptr, m, d_input.ptr, k, &beta, d_output.ptr + ptr_offset, m);
		cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_kernel.ptr, CUDA_R_32F, m, d_input.ptr, CUDA_R_32F, k, &beta, d_output.ptr + ptr_offset, CUDA_R_32F, m);

		// add bias	
		if(hasbias)
		{
			dense_add<<<n, m>>>(m*n, d_bias.ptr, d_output.ptr, ptr_offset);
		}
	}
}



void dense::forward(cudnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, size_t ldc, float alp, float bet)
{

	const float alpha = alp, beta = bet;
	size_t m = d_kernel.shape[1]; 
	size_t k = d_kernel.shape[0];
	size_t n = d_input.shape[0];

	{

		m = d_input.shape[1];
		n = d_kernel.shape[0];
		k = d_kernel.shape[1];
		// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_kernel.ptr, m, d_input.ptr, k, &beta, d_output.ptr, m);
		// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_input.ptr, m, d_kernel.ptr, k, &beta, d_output.ptr, ldc);
		cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_input.ptr, CUDA_R_32F, m, d_kernel.ptr, CUDA_R_32F, k, &beta, d_output.ptr, CUDA_R_32F, ldc);

		if(hasbias)
		{	
			dense_add_conv<<<(d_output.size())/512, 512>>>(d_output.size(), d_bias.ptr, d_output.ptr, ldc);
		}
	}
}


// free host & device memory
dense::~dense()
{
	cublasDestroy ( handle );
}
