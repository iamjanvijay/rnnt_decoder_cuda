#ifndef __DENSE_HPP__
#define __DENSE_HPP__

#include <data_types.hpp>
#include <cudnn.h>
#include<logger.hpp>
#include <cublas_v2.h>

namespace s2t 
{
	using namespace common;

	namespace sys 
	{
		class dense
		{
		private:
			cublasHandle_t handle ;
			gpu_float_array d_kernel;  // this is Rows*Cols [ column major memory layoput ]
			const_gpu_float_array d_bias;  // Bias vector at gpu constant memory

			bool hasbias;

		public:
			noCopy(dense);
			dense();
			/* 
			Alloc memory at device and memcopy the parameters ( shared memory )
			* */
			void init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias);
			void init(const cnpy::NpyArray& h_kernel);
			void operator() (cudnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, size_t offset=0);
			void forward (cudnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, size_t ldc, float alp=1.0, float bet =1.0);

			// free host & device memory
			~dense();
		};
	}
}

#endif
