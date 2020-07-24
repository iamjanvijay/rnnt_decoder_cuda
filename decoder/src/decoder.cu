#include<hparams.hpp>
#include<data_types.hpp>
#include<decoder.hpp>

#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<cassert>
#include<utility> 
#include<queue>
#include<limits>

using namespace s2t::decodernet;
using namespace s2t::sys;
using namespace s2t::common;
using namespace std;

// kernel for decoder compuatations
__global__ void decoder_concat(size_t in1_sz, size_t in2_sz, float* in1, const float* in2)
{
	// concat in2 into in1
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < in2_sz)
	{
		in1[in1_sz+index] = in2[index];
	}
}

// min_first methods
bool min_first::operator()(pair<float, long long> const& pair1, pair<float, long long> const& pair2) 
{ 
    // pair with minimum value of first will be at the top of priority queue
    return pair1.first > pair2.first;
} 

// TrieNode methods
TrieNode::TrieNode()
{
    isCompleteWord = false;
    for (int i = 0; i < letters; ++i)
    {
        children[i] = NULL;
    }
}

TrieNode::~TrieNode()
{
    
}

// Trie methods
Trie::Trie()
{
    root = new TrieNode();
    all_trie_nodes.push_back(root);
}

bool Trie::insert_and_check(vector<size_t>& word) 
{
    /* returns true if word already exists; 
    else returns false and inserts word */
    auto current = root;
    for(int i = 0; i < word.size(); ++i)
    {
        int index = word[i];
        if(!current->children[index])
        {
            current->children[index] = new TrieNode();
            all_trie_nodes.push_back(current->children[index]);
        }
        current = current->children[index];        
    }
    if(current->isCompleteWord)
        return true;
    current->isCompleteWord = true;
    return false;
}

Trie::~Trie()
{
    for(int i=0; i<all_trie_nodes.size(); ++i)
    {
        delete all_trie_nodes[i];
    }
}

// decoder methods
decoder::decoder(size_t p_vocab_size, size_t p_blank_index)
{
    vocab_size = p_vocab_size;
    blank_index = p_blank_index;

    // Read the subword file
    {
        string subword;
        ifstream subwords(hparams::subword_file);
        if(subwords.is_open())
        {
            while(getline(subwords, subword))
            {
                subword_map.push_back(subword);
            }
            subwords.close();
        }
        subword_map.push_back(""); // appending blank symbol at last

        assert(vocab_size==subword_map.size() && "Number of subwords in file and vocab_size do not match!");
        assert(vocab_size==hparams::joint_net_logit_size && "hparams::joint_net_logit_size and vocab_size do not match!");
    }

    // intialise prednet and jointnet
    {
        checkCUDNN(cudnnCreate(&cudnn));
        prednet1.init(cudnn, "");
        jointnet1.init(cudnn, "");
    }

    // initialise the gpu variables
    {
        prednet_out.init(hparams::max_input_size, hparams::pred_net_logit_size);
        enc_pred_concated.init(hparams::max_input_size, hparams::enc_net_logit_size+hparams::pred_net_logit_size); // first 700 enocder, next 700 decoder
        jointnet_out.init(hparams::max_input_size, hparams::joint_net_logit_size);
    }

    // initialise the cpu variables
    {
        log_probs = (float*) malloc(hparams::joint_net_logit_size * sizeof(float));
    }
}

void decoder::operator() (const string& encoder_features_file, size_t beamsize, vector<pair<string, float>>& beams_and_logprobs_out)
{
    auto encoder_features = cnpy::npy_load(encoder_features_file); 
    size_t acoustic_time_steps = encoder_features.shape[0]; // T * 700 file

    // b_heap related data structures
    vector<pair<float*, float*>> next_hiddens;
    vector<data_tuple> data_b;
    priority_queue<pair<float, int>, vector<pair<float, int>>, min_first> b_heap;

    // a_heap realted data structures
    vector<pair<float*, float*>> cur_hiddens; // pointer to h_state, c_state
    vector<bool> is_cur_hidden_useful;
    vector<data_tuple> data_a;
    priority_queue<pair<float, int>, vector<pair<float, int>>, min_first> a_heap;

    // initialse b_heap related data structures before t=0
    float* init_cell_state_h = (float*) calloc(hparams::pred_net_state_h_size * hparams::pred_net_lstm_layers, sizeof(float));
    float* init_cell_state_c = (float*) calloc(hparams::pred_net_state_c_size * hparams::pred_net_lstm_layers, sizeof(float));
    next_hiddens.push_back(make_pair(init_cell_state_h, init_cell_state_c));
    data_tuple init_data_tuple = {"", 0.f, blank_index, 0, {blank_index}};
    data_b.push_back(init_data_tuple);
    b_heap.push(make_pair(0.f, 0));

    for(int t=0; t<acoustic_time_steps; ++t)
    {
        enc_pred_concated.copy(encoder_features.data<float_t>() + hparams::enc_net_logit_size*t, hparams::enc_net_logit_size);

        // boost the probabilities in B
        {

        }

        // delete all for a_heap; 
        {
            // for(int i=0; i<cur_hiddens.size(); i++)
            // {
            //     if(is_cur_hidden_useful[i])
            //         continue;
            //     free(cur_hiddens[i].first);
            //     free(cur_hiddens[i].second);
            // }
            cur_hiddens.clear();
            is_cur_hidden_useful.clear();
            data_a.clear();
            while(a_heap.size()) // reset it
            {
                a_heap.pop();
            }
        }

        // put all data from b_heap in to a_heap and initialise empty b_heap;
        {
            cur_hiddens = next_hiddens;
            data_a = data_b;
            while(b_heap.size())
            {
                pair<float, int> log_prob_data_idx_pair = b_heap.top();
                log_prob_data_idx_pair.first = -log_prob_data_idx_pair.first;
                a_heap.push(log_prob_data_idx_pair);
                b_heap.pop();
            }
            for(int i=0; i<cur_hiddens.size(); i++)
            {
                is_cur_hidden_useful.push_back(false);
            }
            next_hiddens.clear();
            data_b.clear();
        }

        // choose the most probable for a_heap and iterate
        pair<float, int> top_log_prob_data_idx_pair = a_heap.top();
        a_heap.pop();
        size_t top_id_data_a = top_log_prob_data_idx_pair.second;
        float top_log_prob_a = data_a[top_id_data_a].log_prob;
        float bmszth_top_log_prob_b = -numeric_limits<float>::infinity(); 
        Trie trie;

        while(top_log_prob_a!=-numeric_limits<float>::infinity() && bmszth_top_log_prob_b<top_log_prob_a)
        {
            // compute next set of log probablities by calling lm and joint net
            size_t input_symbol = data_a[top_id_data_a].last_decoded_sid;
            pair<float*, float*> prev_states = cur_hiddens[data_a[top_id_data_a].hidden_idx];

            // calls to jointnet and prednet
            prednet1.load_input_state(prev_states.first, prev_states.second);
            prednet1(cudnn, input_symbol, prednet_out);
            decoder_concat<<<1, 1024>>>(700, 700, enc_pred_concated.ptr, prednet_out.ptr); 
            jointnet1(cudnn, enc_pred_concated, jointnet_out);

            // loading state and log_probs in float array
            float* h_cell_state_h = (float*) malloc(hparams::pred_net_state_h_size * hparams::pred_net_lstm_layers * sizeof(float));
            float* h_cell_state_c = (float*) malloc(hparams::pred_net_state_c_size * hparams::pred_net_lstm_layers * sizeof(float));
            prednet1.get_output_state(&h_cell_state_h, &h_cell_state_c);
            size_t log_probs_N = jointnet_out.data_at_host(&log_probs);

            // add blank transition to B
            if(top_log_prob_a+log_probs[blank_index] > bmszth_top_log_prob_b && !trie.insert_and_check(data_a[top_id_data_a].beam_sids)) // and not already in trie:
            {
                data_tuple next_data_tuple = {data_a[top_id_data_a].beam_string, top_log_prob_a + log_probs[blank_index], data_a[top_id_data_a].last_decoded_sid, next_hiddens.size(), data_a[top_id_data_a].beam_sids};
    
                b_heap.push(make_pair(next_data_tuple.log_prob, data_b.size()));
                data_b.push_back(next_data_tuple);
                next_hiddens.push_back(cur_hiddens[data_a[top_id_data_a].hidden_idx]);
                is_cur_hidden_useful[data_a[top_id_data_a].hidden_idx] = true;

                if(b_heap.size()==beamsize+1)
                {
                    b_heap.pop();
                }
                if(b_heap.size()==beamsize)
                {
                    pair<float, int> log_prob_data_idx_pair = b_heap.top();
                    bmszth_top_log_prob_b = data_b[log_prob_data_idx_pair.second].log_prob;
                }
            }     

            // add non-blank transition to A
            for(int i=0; i<301; i++)
            {
                if(i==blank_index || top_log_prob_a+log_probs[i] <= bmszth_top_log_prob_b)
                    continue;

                data_tuple next_data_tuple = {data_a[top_id_data_a].beam_string + subword_map[i], top_log_prob_a + log_probs[i], size_t(i), int(cur_hiddens.size()), data_a[top_id_data_a].beam_sids};
                next_data_tuple.beam_sids.push_back(i);

                a_heap.push(make_pair(-next_data_tuple.log_prob, data_a.size()));
                data_a.push_back(next_data_tuple);
            }
            cur_hiddens.push_back(make_pair(h_cell_state_h, h_cell_state_c));
            is_cur_hidden_useful.push_back(false);  

            // update top_id_data_a and top_log_prob_a
            top_log_prob_a = -numeric_limits<float>::infinity(); 
            if(a_heap.size())
            {
                top_log_prob_data_idx_pair = a_heap.top();
                a_heap.pop();
                top_id_data_a = top_log_prob_data_idx_pair.second;
                top_log_prob_a = data_a[top_id_data_a].log_prob;
            } 
        }
    }

    // dealloc all hiddens floats and log_prob
    // for(int i=0; i<cur_hiddens.size(); i++)
    // {
    //     free(cur_hiddens[i].first);
    //     free(cur_hiddens[i].second);
    // }
    // for(int i=0; i<next_hiddens.size(); i++)
    // {
    //     free(next_hiddens[i].first);
    //     free(next_hiddens[i].second);
    // }
    // cout << "Deallocated hiddens and log_prob!" << endl;

    // write to beams_and_logprobs_out
    while(b_heap.size())
    {
        pair<float, int> log_prob_data_idx_pair = b_heap.top(); b_heap.pop();
        int data_b_idx = log_prob_data_idx_pair.second;
        beams_and_logprobs_out.push_back(make_pair(data_b[data_b_idx].beam_string, data_b[data_b_idx].log_prob));
    }
}

decoder::~decoder()
{
    // de-initialise the cpu variables
    {
        free(log_probs);
    }
}