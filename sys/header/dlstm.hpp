#ifndef __DLSTM_HPP__
#define __DLSTM_HPP__
// Class for forward propagation of LSTM with zoneout in CUDA

#include <cudnn.h>
#include <cublas_v2.h>
#include<data_types.hpp>

namespace s2t 
{
	using namespace common;

	namespace sys 
	{
		class dlstm
		{
		public:
			gpu_float_array d_kernel;
			gpu_float_array d_bias;

			// workspace for lastm execution
			gpu_float_array tmp_h;
			gpu_float_array tmp_i;
			gpu_float_array var1;

			// cublas handle
			cublasHandle_t handle;

			// model parameters
			size_t numLayers;
			size_t hiddenSize;
			size_t inputSize;
			size_t numElements;

			cudaStream_t stream_i;
			cudaStream_t stream_h;

			dim3 blockDim;
			dim3 gridDim;
			
		public:
			dlstm();
			void init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias, 
				size_t numLayers, size_t hiddenSize, size_t inputSize);
			
			void operator()(const gpu_float_array& input, const gpu_float_array& h_data, const gpu_float_array& c_data
				, gpu_float_array& output, gpu_float_array& h_data_out, gpu_float_array& c_data_out, 
				float zoneout_factor_cell = 0.f, float zoneout_factor_outputs = 0.f, float forget_bias = 1.f) ;

			~dlstm();
		};
	}
}
#endif
