// Class for forward propagation of LSTM with zoneout in CUDA

#include <dlstm.hpp>
#include<macros.hpp>
#include<logger.hpp>

using namespace s2t::sys;

__forceinline__ __device__ float d_sigmoidf(float in) {
	return 1.f / (1.f + expf(-in));  
}

__global__ void d_elementWise_fp(int hiddenSize, int miniBatch,
	float *tmp_h, 
	float *tmp_i, 
	float *bias,
	float *h_in,
	float *h_out,
	float *i_out,
	float *c_in,
	float *c_out,
	float zoneout_cell = 0.f,
	float zoneout_outputs = 0.f,
	float forget_bias = 1.f) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int numElements = hiddenSize * miniBatch; 
	if (index >= numElements) return;
	
	int batch = index / hiddenSize;
	int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;   
	
	float g[4];

	for (int i = 0; i < 4; i++) {
		g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
		g[i] += bias[i * hiddenSize + index % hiddenSize] + ((i==2) ? forget_bias : 0); 
	}   
	
	float in_gate     = d_sigmoidf(g[0]);
	float in_gate2    = tanhf(g[1]);
	float forget_gate = d_sigmoidf(g[2]);
	float out_gate    = d_sigmoidf(g[3]);
	
	float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
	
	c_out[index] = (1.f - zoneout_cell) * val + zoneout_cell * c_in[index];
	
	val = out_gate * tanhf(val);                                   

	h_out[index] = (1.f - zoneout_outputs) * val + zoneout_outputs * h_in[index];
	i_out[index] = val;
}

dlstm::dlstm() {}

void dlstm::init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias, 
	size_t numLayers, size_t hiddenSize, size_t inputSize)
{
	this->numLayers = numLayers;
	this->hiddenSize = hiddenSize;
	this->inputSize = inputSize;
	this->numElements = hiddenSize;

	// load kernel
	d_kernel.init(h_kernel.shape);
	cudaMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
	
	// load bias
	d_bias.init(h_bias.shape);
	cudaMemcpy(d_bias.ptr, h_bias.data<float_t>(), d_bias.size()*sizeof(float_t), cudaMemcpyHostToDevice);
	
	checkCUBLAS(cublasCreate(&handle));

	tmp_h.init(4 * numLayers, numElements);
	tmp_i.init(4, numElements);
	var1.init(1,numElements);

	blockDim.x = 256;
	gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;  
}

void dlstm::operator()(const gpu_float_array& input, const gpu_float_array& h_data, const gpu_float_array& c_data
	, gpu_float_array& output, gpu_float_array& h_data_out, gpu_float_array& c_data_out, 
	float zoneout_factor_cell, float zoneout_factor_outputs, float forget_bias) 
{
	float alpha = 1.f;
	float beta  = 0.f;   

	tmp_i.reset();
	tmp_h.reset();
	var1.reset();
	
	checkCUBLAS(cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		4 * hiddenSize, 
		1, 
		inputSize,
		&alpha,
        d_kernel.ptr,   // kernel
        4 * hiddenSize,
        input.ptr,   // data
        inputSize,
        &beta,
        tmp_i.ptr,   // output
        4 * hiddenSize));


	checkCUBLAS(cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		4 * hiddenSize, 
		1, 
		hiddenSize,
		&alpha,
		d_kernel.ptr + (4 * hiddenSize * inputSize), 
		4 * hiddenSize,
        h_data.ptr,  // h_data
        hiddenSize,
        &beta,
        tmp_h.ptr,  // output
        4 * hiddenSize));

	//checkCUDAERROR(cudaDeviceSynchronize());              

	d_elementWise_fp <<< gridDim, blockDim >>> 
	(hiddenSize, 
		1,
		tmp_h.ptr, 
		tmp_i.ptr, 
		d_bias.ptr,
		h_data.ptr,
        h_data_out.ptr, // + numElements,   // h_out
        var1.ptr, // +  inputSize,   // i_out
        c_data.ptr,
        c_data_out.ptr, // + numElements,   // c_out
        zoneout_factor_cell, 
		zoneout_factor_outputs,
		forget_bias); 

	//checkCUDAERROR(cudaDeviceSynchronize());


// layer -2 operation..
	checkCUBLAS(cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		4 * hiddenSize, 
		1, 
		hiddenSize,
		&alpha,
		d_kernel.ptr + (hiddenSize * (hiddenSize + inputSize) * 4),
		4 * hiddenSize,
       var1.ptr, // + inputSize,
       hiddenSize,
       &beta,
       tmp_i.ptr,
       4 * hiddenSize));

	
	checkCUBLAS(cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		4 * hiddenSize, 
		1, 
		hiddenSize,
		&alpha,
		d_kernel.ptr + ((hiddenSize * hiddenSize * 4 + hiddenSize * (hiddenSize + inputSize) * 4)), 
		4 * hiddenSize,
		h_data.ptr +  numElements,
		hiddenSize,
		&beta,
		tmp_h.ptr + (4 * numElements), 
		4 * hiddenSize));

	//checkCUDAERROR(cudaDeviceSynchronize()); 
	d_elementWise_fp <<< gridDim, blockDim >>> 
		(hiddenSize, 1,
		tmp_h.ptr + (4 * numElements), 
		tmp_i.ptr, 
		d_bias.ptr + (4 * hiddenSize),
		h_data.ptr +  numElements,
		h_data_out.ptr +  numElements,   // h_out
        output.ptr, // +  numElements + inputSize,                      //i_out
        c_data.ptr +  numElements,
        c_data_out.ptr +  numElements,    //c_out
        zoneout_factor_cell, 
		zoneout_factor_outputs,
		forget_bias); 

	//checkCUDAERROR(cudaDeviceSynchronize());
}

dlstm::~dlstm() { 
}
