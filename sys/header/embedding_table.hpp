/** 

  --This class searches the embedding vector for an character in embedding table and places the vector into given memory location
  --All operations are done at device only
  --This doesn't create any memory so making free is the responsibility of caller

**/
#ifndef __embedding_hpp__
#define __embedding_hpp__

#pragma once

#include <cudnn.h>
#include <data_types.hpp>
#include<macros.hpp>
#include<cnpy.hpp>

#include <stdio.h>

using namespace s2t::common; 

namespace s2t 
{
	namespace sys 
	{			
		class embedding_table
		{
		private:
			gpu_float_array d_table;  // device memory

		public:
			noCopy(embedding_table);
			embedding_table()		
			{ }

			void init(cnpy::NpyArray h_data)
			{
				d_table.init(h_data.shape);
				cudaMemcpy(d_table.ptr, h_data.data<float_t>(), d_table.size()*sizeof(float_t), cudaMemcpyHostToDevice);
			}
	
			void lookup(cudnnHandle_t& cudnn, const std::vector<size_t>& seq, gpu_float_array& output);
			~embedding_table()
			{ }
		};
	}
}

#endif