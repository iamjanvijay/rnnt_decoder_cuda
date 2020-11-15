import os
import sys
import shutil

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

def getTranspose(input_fn, output_fn):
    k1 = np.load(input_fn)                    
    k1_T = np.ascontiguousarray(k1.T, dtype=np.float32)
    # k1_T.flags['C_CONTIGUOUS']
    np.save(output_fn, k1_T)

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":

    # Reading command line args
    torch_ckpt_path = sys.argv[1] # ../asr/ckpts/rnnt_subword_300_fresh_bz_25/ckpt_16
    output_folder = sys.argv[2] # params/weights

    # Creating model, loading checkpoint and creating output folder
    checkpoint = torch.load(torch_ckpt_path)
    create_folder(output_folder)
    clean_folder(output_folder)

    # Extracting weights
    prednet_wt_names = {'embedding': [], 'lstm_10': [], 'lstm_11': []}
    jointnet_wt_names = {'dense': [], 'dense_1': []}

    # Predicition Network
    prednet_wt_names['embedding'].append(['pred_net_embedding_embeddings:0', checkpoint['model_state']['module.pred_net.embedding.weight'].cpu().numpy()])
    
    prednet_wt_names['lstm_10'].append(['pred_net_lstm_10_kernel:0', checkpoint['model_state']['module.pred_net.rnn.layer_0.weight_ih_l0'].cpu().numpy().T])
    prednet_wt_names['lstm_10'].append(['pred_net_lstm_10_recurrent_kernel:0', checkpoint['model_state']['module.pred_net.rnn.layer_0.weight_hh_l0'].cpu().numpy().T])
    prednet_wt_names['lstm_10'].append(['pred_net_lstm_10_bias:0', checkpoint['model_state']['module.pred_net.rnn.layer_0.bias_ih_l0'].cpu().numpy() + checkpoint['model_state']['module.pred_net.rnn.layer_0.bias_hh_l0'].cpu().numpy()])

    prednet_wt_names['lstm_11'].append(['pred_net_lstm_11_kernel:0', checkpoint['model_state']['module.pred_net.rnn.layer_1.weight_ih_l0'].cpu().numpy().T])
    prednet_wt_names['lstm_11'].append(['pred_net_lstm_11_recurrent_kernel:0', checkpoint['model_state']['module.pred_net.rnn.layer_1.weight_hh_l0'].cpu().numpy().T])
    prednet_wt_names['lstm_11'].append(['pred_net_lstm_11_bias:0', checkpoint['model_state']['module.pred_net.rnn.layer_1.bias_ih_l0'].cpu().numpy() + checkpoint['model_state']['module.pred_net.rnn.layer_1.bias_hh_l0'].cpu().numpy()])

    # Joint Network
    jointnet_wt_names['dense'].append(['joint_net_dense_kernel:0', np.ascontiguousarray(checkpoint['model_state']['module.joint_net.combine.weight'].cpu().numpy().T)])
    jointnet_wt_names['dense'].append(['joint_net_dense_bias:0', checkpoint['model_state']['module.joint_net.combine.bias'].cpu().numpy()])

    jointnet_wt_names['dense_1'].append(['joint_net_dense_1_kernel:0', np.ascontiguousarray(checkpoint['model_state']['module.joint_net.dense.weight'].cpu().numpy().T)])
    jointnet_wt_names['dense_1'].append(['joint_net_dense_1_bias:0', checkpoint['model_state']['module.joint_net.dense.bias'].cpu().numpy()])

    # for weight in model.weights:
    #     name = weight.name
    #     for prednet_wt_name in prednet_wt_names:
    #         if name.startswith(prednet_wt_name):
    #             fname = os.path.join(output_folder, 'pred_net_' + name.replace('/','_'))
    #             prednet_wt_names[prednet_wt_name].append([fname, weight.numpy()])
    #     for jointnet_wt_name in jointnet_wt_names:
    #         if name.startswith(jointnet_wt_name):
    #             fname = os.path.join(output_folder, 'joint_net_' + name.replace('/','_'))
    #             jointnet_wt_names[jointnet_wt_name].append([fname, weight.numpy()])

    # Modify weights
    for prednet_wt_name in prednet_wt_names:
        if prednet_wt_name.startswith('lstm'):
            kernels = []
            bias = []
            final = []
            
            for fname, weight in prednet_wt_names[prednet_wt_name]:
                if 'kernel' in fname:
                    kernels.append([fname, weight])
                else:
                    bias.append([fname, weight])
            
            assert(len(kernels)==2 and len(bias)==1)
            
            # Appending bias to final list
            bias_splits = tf.split(bias[0][1], 4, 0)
            bias_splits[1], bias_splits[2] = bias_splits[2], bias_splits[1]
            bias[0][1] = tf.concat(bias_splits, axis=0)

            final.extend(bias)
            
            # Appending kernel to final list
            if 'recurrent' in kernels[0][0]:
                kernels[0], kernels[1] = kernels[1], kernels[0]
            temp = np.concatenate([kernels[0][1], kernels[1][1]], axis=0)

            # Changes to make it TF 1.x LSTM cell compatible
            temp_splits = tf.split(temp, 4, 1)
            temp_splits[1], temp_splits[2] = temp_splits[2], temp_splits[1]
            temp = tf.concat(temp_splits, axis=1)
            
            final.append([kernels[0][0], temp])

            prednet_wt_names[prednet_wt_name] = final

    # Saving weights
    for prednet_wt_name in prednet_wt_names:
        for fname, weight in prednet_wt_names[prednet_wt_name]:
            np.save(os.path.join(output_folder, fname), weight)
    for jointnet_wt_name in jointnet_wt_names:
        for fname, weight in jointnet_wt_names[jointnet_wt_name]:
            np.save(os.path.join(output_folder, fname), weight)

    # Saving transposed weights
    wt_names = [filename for filename in os.listdir(output_folder) if filename.endswith('.npy')]
    for wt_name in wt_names:
        ip_filename = os.path.join(output_folder, wt_name)
        op_filename = os.path.join(output_folder, wt_name.rstrip('.npy') + '_T.npy')
        getTranspose(ip_filename, op_filename)


# # Checking compatibility of tf2.0 and pytorch - Dense layer
# a = np.random.randn(1, 4, 8).astype(np.float32)
# a_torch = torch.from_numpy(a)
# layer_tf = tf.keras.layers.Dense(16, use_bias=True)
# layer_torch = nn.Linear(8, 16)

# tf_out = layer_tf(a)
# torch_out = layer_torch(a_torch)

# tf_kernel = torch.from_numpy(layer_tf.trainable_weights[0].numpy().T)
# tf_bias = torch.from_numpy(layer_tf.trainable_weights[1].numpy())

# state_dict = layer_torch.state_dict()
# state_dict['weight'] = tf_kernel
# state_dict['bias'] = tf_bias
# layer_torch.load_state_dict(state_dict)

# torch_out = layer_torch(a_torch)

# print("Max abs. diff:", np.max(np.abs(torch_out.detach().numpy()-tf_out)))
# print("done!")






