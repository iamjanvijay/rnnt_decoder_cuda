#ifndef __JOINTNET_HPP__
#define __JOINTNET_HPP__

#pragma once 

#include<memory> 
#include<hparams.hpp>
#include<dense.hpp>
#include<activation.hpp>

namespace s2t
{
	namespace decodernet
	{
		class jointnet
		{
		private:
			sys::dense dense_1;
			sys::dense dense_2;
				
			gpu_float_array var1;
			gpu_float_array var2;

			sys::activation activation_t;

			float one = 1;
			float zero = 0;
			cudnnTensorDescriptor_t tExamples;

		public:
			noCopy(jointnet);
			jointnet();  
			void init(cudnnHandle_t& cudnn, const std::string& base_model_path);  // initialize the jointnet
			void operator() (cudnnHandle_t& cudnn, gpu_float_array& input, gpu_float_array& output);
			~jointnet();  // free all resources
		};
	}
}

#endif
