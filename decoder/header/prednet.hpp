#ifndef __PREDNET_HPP__
#define __PREDNET_HPP__

#pragma once 

#include<memory> 
#include<embedding_table.hpp>
#include<dlstm.hpp>
#include<hparams.hpp>

namespace s2t
{
	namespace decodernet
	{
		struct DLSTMState
		{
			gpu_float_array cell_state_h;
			gpu_float_array cell_state_c;
		};

		class prednet
		{
		private:
			sys::embedding_table embed_t;
			sys::dlstm lstm_t;

			DLSTMState state[2];
				
			gpu_float_array var1;
			gpu_float_array var2;

		public:
			noCopy(prednet);
			prednet();  
			void init(cudnnHandle_t& cudnn, const std::string& base_model_path);  // initialize the prednet
			void reset_input_state();
			void load_input_state(float* h_cell_state_h, float* h_cell_state_c);
			void get_output_state(float** h_cell_state_h, float** h_cell_state_c);
			void operator() (cudnnHandle_t& cudnn, const size_t input_symbol, gpu_float_array& output, bool reverse = false);
			~prednet();  // free all resources
		};
	}
}

#endif
