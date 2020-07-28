import numpy as np
import os

def max_abs_error(arr1, arr2):
    return np.max(np.abs(arr1-arr2))

if __name__ == "__main__":
    base_true_outputs_path = '../params/true_outputs'
    base_outputs_path = '../params/outputs'

    # Test Prediction Network
    cpp_prednet_output = np.load(os.path.join(base_outputs_path, 'cpp_prednet_output.npy'))
    py_prednet_output = np.load(os.path.join(base_true_outputs_path, 'py_prednet_output.npy'))
    print("Prediction Network (logits) MAE:", max_abs_error(cpp_prednet_output, py_prednet_output))

    # Test Joint Network
    cpp_joint_net_dense_2_lsoftmax = np.load(os.path.join(base_outputs_path, 'cpp_joint_net_dense_2_softmax_final.npy'))
    py_joint_net_dense_2_lsoftmax = np.load(os.path.join(base_true_outputs_path, 'py_joint_net_dense_2_softmax.npy'))
    print("Joint Network (log softmax) MAE:", max_abs_error(cpp_joint_net_dense_2_lsoftmax, py_joint_net_dense_2_lsoftmax))