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

			DLSTMState state_buffer[hparams::gpu_states_buffer_size];
			int next_free_state[hparams::gpu_states_buffer_size];
			int state_use_count[hparams::gpu_states_buffer_size];
			int first_free_state_idx;
				
			gpu_float_array var1;
			gpu_float_array var2;

		public:
			noCopy(prednet);
			prednet();  
			void init(cudnnHandle_t& cudnn, const std::string& base_model_path);  // initialize the prednet
			void free_state(int idx);
			void reuse_state(int idx);
			int get_zerod_state();
			int operator() (cudnnHandle_t& cudnn, const size_t input_symbol, gpu_float_array& output, int input_state_idx, bool save_state = true);
			~prednet();  // free all resources
		};
	}
}

#endif
