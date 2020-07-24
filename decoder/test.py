import numpy as np
import os
import tensorflow as tf

base_path = '/home/janvijay.singh/ASR/rnnt_decoder/tts/cpp/decoder/'

# Prediction Network
cpp_embedding_1 = np.load(os.path.join(base_path, 'cpp_embedding_1.npy'))
cpp_c_input = np.load(os.path.join(base_path, 'cpp_c_input.npy'))
cpp_c_output = np.load(os.path.join(base_path, 'cpp_c_output.npy'))
cpp_h_input = np.load(os.path.join(base_path, 'cpp_h_input.npy'))
cpp_h_output = np.load(os.path.join(base_path, 'cpp_h_output.npy'))
cpp_prednet_output = np.load(os.path.join(base_path, 'cpp_prednet_output.npy'))

py_embedding_1 = np.load(os.path.join(base_path, 'py_embedding_1.npy'))
py_c_0_input = np.load(os.path.join(base_path, 'py_c_0_input.npy'))
py_c_0_output = np.load(os.path.join(base_path, 'py_c_0_output.npy'))
py_c_1_input = np.load(os.path.join(base_path, 'py_c_1_input.npy'))
py_c_1_output = np.load(os.path.join(base_path, 'py_c_1_output.npy'))
py_h_0_input = np.load(os.path.join(base_path, 'py_h_0_input.npy'))
py_h_0_output = np.load(os.path.join(base_path, 'py_h_0_output.npy'))
py_h_1_input = np.load(os.path.join(base_path, 'py_h_1_input.npy'))
py_h_1_output = np.load(os.path.join(base_path, 'py_h_1_output.npy'))
py_prednet_output = np.load(os.path.join(base_path, 'py_prednet_output.npy'))

# Joint Network
cpp_joint_net_dense_1 = np.load(os.path.join(base_path, 'cpp_joint_net_dense_1.npy.npy'))
cpp_joint_net_dense_1_relu = np.load(os.path.join(base_path, 'cpp_joint_net_dense_1_relu.npy.npy'))
cpp_joint_net_dense_2 = np.load(os.path.join(base_path, 'cpp_joint_net_dense_2.npy'))
cpp_joint_net_dense_2_softmax = np.load(os.path.join(base_path, 'cpp_joint_net_dense_2_softmax.npy.npy'))
cpp_joint_net_input_concatenated_features_gpu = np.load(os.path.join(base_path, 'cpp_joint_net_input_concatenated_features_gpu.npy'))
cpp_joint_net_input_concatenated_features = np.load(os.path.join(base_path, 'cpp_joint_net_input_concatenated_features.npy'))

py_joint_net_dense_1_relu = np.load(os.path.join(base_path, 'py_joint_net_dense_1_relu.npy'))
py_joint_net_dense_2 = np.load(os.path.join(base_path, 'py_joint_net_dense_2.npy'))
py_joint_net_dense_2_softmax = np.load(os.path.join(base_path, 'py_joint_net_dense_2_softmax.npy'))
py_joint_net_input_encoder_features = np.load(os.path.join(base_path, 'py_joint_net_input_encoder_features.npy'))
py_joint_net_input_prediction_features = np.load(os.path.join(base_path, 'py_joint_net_input_prediction_features.npy'))

print("PREDNET:\n")

print("INPUT EMBEDDING:")
print("Input", np.max(np.abs(cpp_embedding_1-py_embedding_1)))

print("INPUT C:")

print("0", np.max(np.abs(cpp_c_input[0]-py_c_0_input)))
print("1", np.max(np.abs(cpp_c_input[1]-py_c_1_input)))

print("INPUT H:")

print("0", np.max(np.abs(cpp_h_input[0]-py_h_0_input)))
print("1", np.max(np.abs(cpp_h_input[1]-py_h_1_input)))

print("OUTPUT C:")

print("0", np.max(np.abs(cpp_c_output[0]-py_c_0_output)))
print("1", np.max(np.abs(cpp_c_output[1]-py_c_1_output)))

print("OUTPUT H:")

print("0", np.max(np.abs(cpp_h_output[0]-py_h_0_output)))
print("1", np.max(np.abs(cpp_h_output[1]-py_h_1_output)), "\n")

print("JOINTNET:\n")

py_ef_pf = np.concatenate([py_joint_net_input_encoder_features, py_joint_net_input_prediction_features], axis=-1)

print("INPUT:")
print("Max abs", np.max(np.abs(py_ef_pf-cpp_joint_net_input_concatenated_features)))

print("GPU INPUT:")
print("Max abs", np.max(np.abs(py_ef_pf-cpp_joint_net_input_concatenated_features_gpu)))

print("DENSE 1 RELU:")
print("Max abs", np.max(np.abs(py_joint_net_dense_1_relu-cpp_joint_net_dense_1_relu)))

print("DENSE 2:")
print("Max abs", np.max(np.abs(py_joint_net_dense_2-cpp_joint_net_dense_2)))

print("DENSE 2 SOFTMAX:")
print("Max abs", np.max(np.abs(py_joint_net_dense_2_softmax-cpp_joint_net_dense_2_softmax)))

print("Done")



