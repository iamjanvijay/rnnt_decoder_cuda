#include<jointnet.hpp>
#include<hparams.hpp>
#include<data_types.hpp>

#include<iostream>
#include<vector>
#include<string>
#include<logger.hpp>
// #include<utils.hpp>
#include<cublas_v2.h>


using namespace s2t::decodernet;
using namespace s2t::common;
using namespace std;

jointnet::jointnet()
{}

void jointnet::init(cudnnHandle_t& cudnn, const std::string& base_model_path)
{
	size_t dense_1_hidden_size = 0;
	size_t dense_2_hidden_size = 0;

	// initialize dense_1
	{
		auto kernel_weight = cnpy::npy_load(base_model_path + hparams::joint_net_dense_0_kernel); 
		auto bias_weight = cnpy::npy_load(base_model_path + hparams::joint_net_dense_0_bias);

        dense_1.init(kernel_weight, bias_weight);
        
		dense_1_hidden_size = kernel_weight.shape[1]; 
		
		// cout << "dense_1_hidden_size: " << dense_1_hidden_size << endl;
	}
	
	// initialise relu activation layer
	{
		activation_t.init(1, dense_1_hidden_size, 1, 1, CUDNN_ACTIVATION_RELU);
	}
    
	// initialize dense_2
	{
		auto kernel_weight = cnpy::npy_load(base_model_path + hparams::joint_net_dense_1_kernel); 
		auto bias_weight = cnpy::npy_load(base_model_path + hparams::joint_net_dense_1_bias);

		dense_2.init(kernel_weight, bias_weight);

		dense_2_hidden_size = kernel_weight.shape[1]; 
	}

	// intitlaize gpu variables
	{
		var1.init(hparams::max_input_size, dense_1_hidden_size);
		// var2.init(hparams::max_input_size, dense_2_hidden_size);
		cudnnCreateTensorDescriptor(&tExamples);
		cudnnSetTensor4dDescriptor(tExamples, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, dense_2_hidden_size, 1, 1);
	}	
}


void jointnet::operator() (cudnnHandle_t& cudnn, gpu_float_array& input, gpu_float_array& output)
{
    // reset and reshape the Vars based on input size
	var1.reset();
	var1.reshape(1, var1.shape[1]);
	// var2.reset();
	// var2.reshape(1, var2.shape[1]);

	dense_1(cudnn, input, var1);
	// log_e("dense 1 output", var1.log("cpp_joint_net_dense_1.npy.npy"));
	activation_t(cudnn, var1);
	// log_e("dense 1 relu output", var1.log("cpp_joint_net_dense_1_relu.npy.npy"));

	dense_2(cudnn, var1, output);
	// log_e("dense 2 output", output.log("cpp_joint_net_dense_2.npy"));

	// cout << output.shape[1] << endl;

	cudnnStatus_t status = cudnnSoftmaxForward(cudnn, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE, &one, tExamples, output.ptr, &zero, tExamples, output.ptr);
	
	// cout << CUDNN_STATUS_BAD_PARAM << " " << CUDNN_STATUS_SUCCESS << " : " << status << endl;
	// log_e("dense 2 softmax output", output.log("cpp_joint_net_dense_2_softmax.npy.npy"));
}

jointnet::~jointnet()
{}