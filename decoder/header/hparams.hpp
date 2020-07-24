#ifndef _HPARAMS_HPP__
#define _HPARAMS_HPP__

#pragma once 

#include<string>
#include <cstdlib>

namespace s2t
{
	namespace decodernet
	{
		namespace hparams
		{
			using namespace std;

			static const string base_param_folder = "../weights/"; 
			static const string base_input_folder = "./"; 

			// encoder network parameters
			
				static const size_t enc_net_logit_size = 700;
			
			// prediction network parameters
			
				// embedding table
				static const string pred_net_embedding = base_param_folder + "pred_net_embedding_embeddings:0.npy";

				// lstm layers
				static const size_t pred_net_lstm_layers = 2;
				static const string pred_net_lstm_0_kernel = base_param_folder + "pred_net_lstm_10_kernel:0.npy";
				static const string pred_net_lstm_0_bias = base_param_folder + "pred_net_lstm_10_bias:0.npy";
				static const string pred_net_lstm_1_kernel = base_param_folder + "pred_net_lstm_11_kernel:0.npy";
				static const string pred_net_lstm_1_bias = base_param_folder + "pred_net_lstm_11_bias:0.npy";

				// other parameters
				static const size_t pred_net_logit_size = 700;
				static const size_t pred_net_state_h_size = 700;
				static const size_t pred_net_state_c_size = 700;
				static const size_t max_input_size = 1;

			// joint network parameters

				// dense_1
				static const string joint_net_dense_0_kernel = base_param_folder + "joint_net_dense_kernel:0.npy";
				static const string joint_net_dense_0_bias = base_param_folder + "joint_net_dense_bias:0.npy";

				// dense_2
				static const string joint_net_dense_1_kernel = base_param_folder + "joint_net_dense_1_kernel:0.npy";
				static const string joint_net_dense_1_bias = base_param_folder + "joint_net_dense_1_bias:0.npy";

				// joint network inputs
				static const string joint_net_encoder_feats = base_input_folder + "py_joint_net_input_encoder_features.npy";
				static const string joint_net_prediction_feats = base_input_folder + "py_joint_net_input_prediction_features.npy";
			
				// other parameters
				static const size_t joint_net_logit_size = 301;

			// decoder specific parameters

				static const string subword_file = base_input_folder + "subword_list.txt";
				static const string output_beams_logprobs_file = base_input_folder + "beam_and_logprobs.txt";

		}
	}
}

#endif
