import tensorflow as tf
import numpy as np
import os

type_test = 2 # 1 for comparing keras lstm and tf 1.x lstm | 2 for using the cpp weights in tf 1.x lstm

if type_test==1:
    tf.random.set_seed(0)
    num_units = 700
    input_sz = 512

    inputs = tf.random.normal([1, 1, input_sz])

    rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units, forget_bias=0.0) # changing forget bias to 0
    multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell])
    layer_2 = tf.keras.layers.LSTM(num_units, kernel_initializer='glorot_uniform', time_major=False, return_sequences=True, return_state=False)

    i_s = multi_rnn_cell.zero_state(1, dtype=tf.float32)

    outputs, s = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=inputs, initial_state=i_s, dtype=tf.float32)
    out_2 = layer_2(inputs)

    w = [x.numpy() for x in multi_rnn_cell.weights]

    # Making changes to kernel
    w_0_split = tf.split(w[0], 4, 1)
    w_0_split[1], w_0_split[2] = w_0_split[2], w_0_split[1]
    w[0] = tf.concat(w_0_split, axis=1)

    w = [w[0][:input_sz, :], w[0][input_sz:, :], w[1]]

    layer_2.set_weights(w)

    out_2 = layer_2(inputs, initial_state=None)

    print(np.mean(np.square(out_2 - outputs)))
else:
    num_units = 700
    input_sz = 512

    rnn_cell_1 = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units, forget_bias=0.0)
    rnn_cell_2 = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units, forget_bias=0.0)
    multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell_1, rnn_cell_2])

    i_s = multi_rnn_cell.zero_state(1, dtype=tf.float32)
    rand_inputs = tf.random.normal([1, 1, input_sz])
    rand_outputs, s = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=rand_inputs, initial_state=i_s, dtype=tf.float32)

    # load weights
    base_path = 'numpy_weights'
    wt_1_kernel, wt_1_bias = 'pred_net_lstm_10_kernel:0.npy', 'pred_net_lstm_10_bias:0.npy'
    wt_2_kernel, wt_2_bias = 'pred_net_lstm_11_kernel:0.npy', 'pred_net_lstm_11_bias:0.npy'
    rnn_cell_1.set_weights([np.load(os.path.join(base_path, wt_1_kernel)), np.load(os.path.join(base_path, wt_1_bias))])
    rnn_cell_2.set_weights([np.load(os.path.join(base_path, wt_2_kernel)), np.load(os.path.join(base_path, wt_2_bias))])

    # load input and run
    base_path = '/home/janvijay.singh/ASR/rnnt_decoder/tts/cpp/pred_network/'
    embedding_1 = np.load(os.path.join(base_path, 'py_embedding_1.npy'))
    output, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=np.reshape(embedding_1, (1 ,1, input_sz)), initial_state=i_s, dtype=tf.float32)

    tf_2_output = np.load(os.path.join(base_path, 'py_prednet_output.npy'))
    cpp_output = np.load(os.path.join(base_path, 'cpp_prednet_output.npy'))

    print("CPP Difference:", np.mean(np.square(cpp_output-output)), np.max(np.abs(cpp_output-output)))
    print("TF 2 Model Difference:", np.mean(np.square(output-tf_2_output)), np.max(np.abs(output-tf_2_output)))

    print("Here")

    
