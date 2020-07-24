#ifndef __DECODER_HPP__
#define __DECODER_HPP__

#pragma once 

#include<hparams.hpp>
#include<vector>
#include<string>
#include<prednet.hpp>
#include<jointnet.hpp>

using namespace std;

namespace s2t
{
    namespace decodernet
    {
        struct data_tuple
        {
            string beam_string; // can change this to long long beam_string_idx;
            float log_prob;
            size_t last_decoded_sid;
            int hidden_idx;
            vector<size_t> beam_sids;
        };

        struct min_first
        { 
            bool operator()(pair<float, long long> const& pair1, pair<float, long long> const& pair2);
        }; 

        class TrieNode
        {    

        public:    
            static const int letters = hparams::joint_net_logit_size;
            TrieNode* children[letters];
            bool isCompleteWord;
            TrieNode();
            ~TrieNode();
        };

        class Trie 
        {

        private:
            TrieNode* root; 
            vector<TrieNode*> all_trie_nodes;

        public:
            Trie();
            bool insert_and_check(vector<size_t>& word);
            ~Trie();
        };

        class decoder
        {

        private:
            size_t vocab_size;
            size_t blank_index;
            vector<string> subword_map;
            // std::vector<bool> is_space_subword;
            // use https://github.com/Tessil/hat-trie to create trie based lexicon

            cudnnHandle_t cudnn;
            prednet prednet1;
            jointnet jointnet1;

            // gpu variables used while decoding
            gpu_float_array prednet_out;
            gpu_float_array enc_pred_concated;
            gpu_float_array jointnet_out;

            // cpu variables used while decoding
            float* log_probs;

        public:
            decoder(size_t p_vocab_size, size_t p_blank_index);  
            void operator() (const string& encoder_features_file, size_t beamsize, vector<pair<string, float>>& beams_and_logprobs_out);
            ~decoder(); // free all resources
        };
    }
}


#endif
