import os
import sys
import shutil

import numpy as np
import tensorflow as tf

sys.path.append('/home/janvijay.singh/ASR/asr_abhinav_subword_300_30_june')
from utils import objdict, checkpoint
from models import create_model

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
    model_num = sys.argv[1]
    output_folder = sys.argv[2]

    # Creating model, loading checkpoint and creating output folder
    checkpoint_path = '/shared/abhinav.goyal/s2t/rnnt_data/ckpt-' + str(model_num)
    model, _ = create_model('rnnt', objdict({'batch_size': 2, 'decoding': False}), build=True)
    checkpoint.load_checkpoint(checkpoint_path, model)
    create_folder(output_folder)
    clean_folder(output_folder)

    # Extracting weights
    prednet_wt_names = {'embedding': [], 'lstm_10': [], 'lstm_11': []}
    jointnet_wt_names = {'dense': [], 'dense_1': []}
    for weight in model.weights:
        name = weight.name
        for prednet_wt_name in prednet_wt_names:
            if name.startswith(prednet_wt_name):
                fname = os.path.join(output_folder, 'pred_net_' + name.replace('/','_'))
                prednet_wt_names[prednet_wt_name].append([fname, weight.numpy()])
        for jointnet_wt_name in jointnet_wt_names:
            if name.startswith(jointnet_wt_name):
                fname = os.path.join(output_folder, 'joint_net_' + name.replace('/','_'))
                jointnet_wt_names[jointnet_wt_name].append([fname, weight.numpy()])

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
            np.save(fname, weight)
    for jointnet_wt_name in jointnet_wt_names:
        for fname, weight in jointnet_wt_names[jointnet_wt_name]:
            np.save(fname, weight)

    # Saving transposed weights
    wt_names = [filename for filename in os.listdir(output_folder) if filename.endswith('.npy')]
    for wt_name in wt_names:
        ip_filename = os.path.join(output_folder, wt_name)
        op_filename = os.path.join(output_folder, wt_name.rstrip('.npy') + '_T.npy')
        getTranspose(ip_filename, op_filename)