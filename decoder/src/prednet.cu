#include<prednet.hpp>
#include<jointnet.hpp>
#include<hparams.hpp>
#include<data_types.hpp>

#include<iostream>
#include<vector>
#include<string>
#include<logger.hpp>

using namespace s2t::decodernet;
using namespace s2t::common;
using namespace std;

prednet::prednet()
{

}

void prednet::init(cudnnHandle_t& cudnn, const std::string& base_model_path)
{
	// initialize embedding_table
	size_t embedding_sz = 0;
	size_t lstm_hidden_size = 0;

	{
		auto arr = cnpy::npy_load(hparams::pred_net_embedding);
		embed_t.init(arr);
		embedding_sz = arr.shape[1];
	}

	size_t lstm_input_size = embedding_sz;

	// intitalize lstm_t
	{
		auto kernel_weight_0 = cnpy::npy_load(hparams::pred_net_lstm_0_kernel); 
		auto bias_weight_0 = cnpy::npy_load(hparams::pred_net_lstm_0_bias); 

		auto kernel_weight_1 = cnpy::npy_load(hparams::pred_net_lstm_1_kernel); 
		auto bias_weight_1= cnpy::npy_load(hparams::pred_net_lstm_1_bias); 

		kernel_weight_0.merge(kernel_weight_1);
		bias_weight_0.merge(bias_weight_1);

		lstm_hidden_size = kernel_weight_0.shape[1] / 4; 

		lstm_t.init(kernel_weight_0, bias_weight_0, hparams::pred_net_lstm_layers, lstm_hidden_size, lstm_input_size);
	}

	// initialize DLSTM state
	for(size_t i=0; i<hparams::gpu_states_buffer_size; ++i)
	{
		state_buffer[i].cell_state_h.init(hparams::pred_net_lstm_layers, lstm_hidden_size);
		state_buffer[i].cell_state_c.init(hparams::pred_net_lstm_layers, lstm_hidden_size);
		state_buffer[i].cell_state_h.reset();
		state_buffer[i].cell_state_c.reset();		

		next_free_state[i] = i+1;
	}
	first_free_state_idx = 0;

	// intitlaize gpu variables
	{
		var1.init(hparams::max_input_size, embedding_sz);
	}	
}

int prednet::get_zerod_state()
{
	assert(first_free_state_idx!=hparams::gpu_states_buffer_size && "Increase DLSTM buffer state size!");

	int state_return_idx = first_free_state_idx;
	state_buffer[first_free_state_idx].cell_state_h.reset();
	state_buffer[first_free_state_idx].cell_state_c.reset();	
	state_use_count[first_free_state_idx] = 0;
	first_free_state_idx = next_free_state[first_free_state_idx];
	return state_return_idx;
}

void prednet::free_state(int idx)
{
	--state_use_count[idx];
	if(state_use_count[idx]<=0)
	{
		next_free_state[idx] = first_free_state_idx;
		first_free_state_idx = idx;
	}
}

void prednet::reuse_state(int idx)
{
	++state_use_count[idx];
}

int prednet::operator() (cudnnHandle_t& cudnn, const size_t input_symbol, gpu_float_array& output, int input_state_idx, int output_state_idx)
{
	assert(first_free_state_idx!=hparams::gpu_states_buffer_size && "Increase DLSTM buffer state size!");

	// reset and reshape the Vars based on input size
	var1.reset();
	var1.reshape(1, var1.shape[1]);

	// get embeddings
	embed_t.lookup(cudnn, {input_symbol}, var1);

	int state_return_idx = (output_state_idx==-1)?first_free_state_idx:output_state_idx;
	// run lstms	
	lstm_t(var1, state_buffer[input_state_idx].cell_state_h, state_buffer[input_state_idx].cell_state_c,
		output, state_buffer[state_return_idx].cell_state_h, state_buffer[state_return_idx].cell_state_c, 
		0 /*zoneout factor cell*/, 0 /*zoneout factor outputs*/, 0 /*forget bias*/);

	if(output_state_idx==-1) // save the state on state_return_idx
	{
		state_use_count[state_return_idx] = 0;
		first_free_state_idx = next_free_state[state_return_idx];
	}
	return state_return_idx; 
}

prednet::~prednet()
{

}