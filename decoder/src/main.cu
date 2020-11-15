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
	prednet1(cudnn, input_symbol, output, input_state_idx);	
	prednet1.free_state(input_state_idx);

	string filename = hparams::base_output_folder + "cpp_prednet_output.npy";
	log_e("pred net output", output.log(filename));
}

void testjointnet(cudnnHandle_t& cudnn)
{
	auto encoder_features = cnpy::npy_load(hparams::joint_net_encoder_feats); 
	auto decoder_features = cnpy::npy_load(hparams::joint_net_prediction_feats); 
	encoder_features.merge(decoder_features);

	gpu_float_array jointnet_input;
	jointnet_input.init(encoder_features.shape);
	cudaMemcpy(jointnet_input.ptr, encoder_features.data<float_t>(), jointnet_input.size()*sizeof(float_t), cudaMemcpyHostToDevice);
	jointnet_input.reshape(1, 1400);

	jointnet jointnet1;
	jointnet1.init(cudnn, "");

	gpu_float_array output;
	output.init(hparams::max_input_size, 301);
	jointnet1(cudnn, jointnet_input, output);

	string filename =  hparams::base_output_folder + "cpp_joint_net_dense_2_softmax_final.npy";
	log_e("joint net output softmax output", output.log(filename));
}

void testdecoder()
{
	const string encoder_features_file = hparams::base_input_folder + "encoder_features.npy";
	size_t beamsize = 1000;
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
			outfile << fixed << setprecision(8) << beams_and_logprobs[i].second << "\n";
		}
		outfile.close();
		cout << "Finised writing output beams!" << endl;
	}
	else
	{
		cout << "Couldn't open output file!\n";
	}
}

void decodemultiple(string metadata_path, int begin_index, int end_index, size_t beamsize, size_t vocab_size)
{
	// size_t beamsize = 100;
	// size_t vocab_size = 5001;
	size_t blank_index = vocab_size-1;	

	cout << "Vocab Size: " << vocab_size << endl;
	cout << "Blank Index: " << blank_index << endl;

	string encoder_features_file;
	decoder decoder1(vocab_size, blank_index);

	auto total_time = 0.0;
	int count = 0;
	ifstream metadata(metadata_path);
	if(metadata.is_open())
	{
		while(getline(metadata, encoder_features_file))
		{
			if(count<begin_index || count>end_index)
			{
				++count;
				continue;
			}

			string encoder_features_file_path = "../data/inputs/" + encoder_features_file;
			string output_beams_logprobs_file = "../data/outputs/" + encoder_features_file + ".txt";
			// cout << encoder_features_file_path << " Input file!" << endl;
			// cout << output_beams_logprobs_file << " Output file!" << endl;
			vector<pair<string, float>> beams_and_logprobs;

			auto time_start = high_resolution_clock::now();
			decoder1(encoder_features_file_path, beamsize, beams_and_logprobs); 
			auto time_stop = high_resolution_clock::now(); 
			auto time_duration = duration_cast<milliseconds>(time_stop - time_start); 
			total_time += time_duration.count();
			// cout << "TIME ELAPSED IN DECODING (" << count << "): " << time_duration.count() << endl; 

			ofstream outfile(output_beams_logprobs_file);
			if (outfile.is_open())
			{
				for(int i=int(beams_and_logprobs.size())-1; i>=0; i--)
				{
					outfile << beams_and_logprobs[i].first << "\t";
					outfile << fixed << setprecision(16) << beams_and_logprobs[i].second << "\n";
				}
				outfile.close();
				++count;
			}
			else
			{
				cout << "Couldn't open output file!\n";
			}
		}
		metadata.close();
	}
	else
	{
		cout << "Couldn't open metadata file!\n";
	}
	cout << "Average time for decoding: " << total_time / (end_index-begin_index+1) << endl;
}

int main(int argc, char *argv[])
{
	// create a cuda handle
	cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));

	// testprednet(cudnn);
	// testjointnet(cudnn);
	// testdecoder();

	assert(argc==6 && "pararm 1: path to metadata; pararm 2: begin_index; pararm 3: end_index; param 4: beamsize; param 5 vocabsize;");
	string metadata_path = argv[1]; 
	int begin_index = stoi(argv[2]), end_index = stoi(argv[3]);
	size_t beamsize = stoi(argv[4]), vocab_size = stoi(argv[5]); // vocab size 301 or 5001
	cout << "Loading weights from: " << s2t::decodernet::hparams::base_param_folder << "\n";
	decodemultiple(metadata_path, begin_index, end_index, beamsize, vocab_size);
	cudnnDestroy(cudnn);
	return 0;
}
// #endif


