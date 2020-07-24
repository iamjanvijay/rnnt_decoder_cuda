#include<prednet.hpp>
#include<jointnet.hpp>
#include<hparams.hpp>
#include<data_types.hpp>

#include<iostream>
#include<vector>
#include<string>
#include<logger.hpp>
// #include<utils.hpp>

using namespace s2t::decodernet;
using namespace s2t::common;
using namespace std;

prednet::prednet()
{}

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

		// cnpy::npy_save("cpp_kernel_0.npy", kernel_weight_0.as_vec<float>());
		// cnpy::npy_save("cpp_bias_0.npy", bias_weight_0.as_vec<float>());

		auto kernel_weight_1 = cnpy::npy_load(hparams::pred_net_lstm_1_kernel); 
		auto bias_weight_1= cnpy::npy_load(hparams::pred_net_lstm_1_bias); 

		// cnpy::npy_save("cpp_kernel_1.npy", kernel_weight_1.as_vec<float>());
		// cnpy::npy_save("cpp_bias_1.npy", bias_weight_1.as_vec<float>());

		kernel_weight_0.merge(kernel_weight_1);
		bias_weight_0.merge(bias_weight_1);

		lstm_hidden_size = kernel_weight_0.shape[1] / 4; 

		lstm_t.init(kernel_weight_0, bias_weight_0, hparams::pred_net_lstm_layers, lstm_hidden_size, lstm_input_size);
	}

	// initialize DLSTM state
	for(size_t i=0;i<2;i++)
	{
		state[i].cell_state_h.init(hparams::pred_net_lstm_layers, lstm_hidden_size);
		state[i].cell_state_c.init(hparams::pred_net_lstm_layers, lstm_hidden_size);
		
		state[i].cell_state_h.reset();
		state[i].cell_state_c.reset();		

	}

	// intitlaize gpu variables
	{
		var1.init(hparams::max_input_size, embedding_sz);
		var2.init(hparams::max_input_size, embedding_sz);
	}	
}

void prednet::reset_input_state()
{
	size_t oldSI = 0;
	state[oldSI].cell_state_h.reset();
	state[oldSI].cell_state_c.reset();	
}

void prednet::load_input_state(float* h_cell_state_h, float* h_cell_state_c)
{
	size_t oldSI = 0;
	state[oldSI].cell_state_h.copy(h_cell_state_h, hparams::pred_net_lstm_layers * hparams::pred_net_state_h_size);
	state[oldSI].cell_state_c.copy(h_cell_state_c, hparams::pred_net_lstm_layers * hparams::pred_net_state_c_size);
}

void prednet::get_output_state(float** h_cell_state_h, float** h_cell_state_c)
{
	// assumes that params have already been allocated sufficient memory
	size_t newSI = 1; 
	size_t state_h_N = state[newSI].cell_state_h.data_at_host(h_cell_state_h);
	size_t state_c_N = state[newSI].cell_state_c.data_at_host(h_cell_state_c);
}

void prednet::operator() (cudnnHandle_t& cudnn, const size_t input_symbol, gpu_float_array& output, bool reverse)
{
	// reset and reshape the Vars based on input size
	var1.reset();
	var1.reshape(1, var1.shape[1]);

	// var2.reset();
	// var2.reshape(seq.size(), var1.shape[1]);

	// get embeddings
	embed_t.lookup(cudnn, {input_symbol}, var1);

	// log_e("embedding 1", var1.log("cpp_embedding_1.npy"));
	
	size_t oldSI = 0;
	size_t newSI = 1; 
	if(reverse)
	{
		oldSI = 1;
		newSI = 0;
	}

	// run lstms
	lstm_t(var1, state[oldSI].cell_state_h, state[oldSI].cell_state_c,
		output, state[newSI].cell_state_h, state[newSI].cell_state_c, 
		0 /*zoneout factor cell*/, 0 /*zoneout factor outputs*/, 0 /*forget bias*/);

	// log_e("h input", state[oldSI].cell_state_h.log("cpp_h_input.npy"));
	// log_e("c input", state[oldSI].cell_state_c.log("cpp_c_input.npy" /*filename of numpy file dumped*/));

	// log_e("h output", state[newSI].cell_state_h.log("cpp_h_output.npy"));
	// log_e("c output", state[newSI].cell_state_c.log("cpp_c_output.npy" /*filename of numpy file dumped*/));
	
}

prednet::~prednet()
{

}