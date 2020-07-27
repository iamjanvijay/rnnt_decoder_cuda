//# define NDEBUG // switch off all the assert calls. 
//#undef NDEBUG
// #ifdef ONLY_T2

#include<prednet.hpp>
#include<jointnet.hpp>
#include<decoder.hpp>

#include<logger.hpp>

#include<data_types.hpp>
#include<cnpy.hpp>
#include<string>
#include<vector>
#include<chrono>
#include<unistd.h>
#include<fstream>
#include<iomanip>

using namespace s2t;
using namespace std;
using namespace s2t::common;
using namespace s2t::decodernet;
using namespace std::chrono;

void testprednet(cudnnHandle_t& cudnn)
{
	// testing prednet
	size_t input_symbol = 1;

	prednet prednet1;
	prednet1.init(cudnn, "");

	gpu_float_array output;
	output.init(hparams::max_input_size, 700);
	int input_state_idx = prednet1.get_zerod_state();
	prednet1.reuse_state(input_state_idx);
	prednet1(cudnn, input_symbol, output, input_state_idx, false /*save_state*/);	
	prednet1.free_state(input_state_idx);

	string filename = "cpp_prednet_output.npy";
	log_e("pred net output", output.log(filename));
}

void testjointnet(cudnnHandle_t& cudnn)
{
	auto encoder_features = cnpy::npy_load(hparams::joint_net_encoder_feats); 
	auto decoder_features = cnpy::npy_load(hparams::joint_net_prediction_feats); 
	encoder_features.merge(decoder_features);

	cnpy::npy_save("cpp_joint_net_input_concatenated_features.npy", encoder_features.as_vec<float>());

	gpu_float_array jointnet_input;
	jointnet_input.init(encoder_features.shape);
	cudaMemcpy(jointnet_input.ptr, encoder_features.data<float_t>(), jointnet_input.size()*sizeof(float_t), cudaMemcpyHostToDevice);
	jointnet_input.reshape(1, 1400);

	log_e("joint net gpu input", jointnet_input.log("cpp_joint_net_input_concatenated_features_gpu.npy"));

	jointnet jointnet1;
	jointnet1.init(cudnn, "");

	gpu_float_array output;
	output.init(hparams::max_input_size, 301);
	jointnet1(cudnn, jointnet_input, output);

	string filename = "cpp_joint_net_dense_2_softmax_final.npy";
	log_e("joint net output softmax output", output.log(filename));
}

void testdecoder()
{
	const string encoder_features_file = hparams::base_input_folder + "encoder_features.npy";
	size_t beamsize = 100;
	size_t vocab_size = 301;
	size_t blank_index = 300;
	decoder decoder1(vocab_size, blank_index);
	vector<pair<string, float>> beams_and_logprobs;
	
	// calling the decoder
	auto time_start = high_resolution_clock::now();
	decoder1(encoder_features_file, beamsize, beams_and_logprobs); 
    auto time_stop = high_resolution_clock::now(); 
    auto time_duration = duration_cast<milliseconds>(time_stop - time_start); 
    cout << "TIME ELAPSED IN DECODING: " << time_duration.count() << endl; 

	assert(beamsize==beams_and_logprobs.size() && "Number of beams returned and beamsize do not match!");

	// write beams to output file
	ofstream outfile(hparams::output_beams_logprobs_file);
	if (outfile.is_open())
	{
		for(int i=0; i<beams_and_logprobs.size(); i++)
		{
			outfile << beams_and_logprobs[i].first << "\t";
			outfile << fixed << setprecision(4) << beams_and_logprobs[i].second << "\n";
		}
		outfile.close();
		cout << "Finised writing output beams!" << endl;
	}
	else
	{
		cout << "Couldn't open output file!\n";
	}
}

int main()
{
	// create a cuda handle
	cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));

	// testprednet(cudnn);
	// testjointnet(cudnn);
	testdecoder();

	cudnnDestroy(cudnn);
	return 0;
}
// #endif


