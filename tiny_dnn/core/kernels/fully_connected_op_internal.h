/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_internal(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

	std::cout << std::endl << "-- Forward in: " << in.size()<< std::endl;
	for (size_t i = 0; i < in.size(); i++) {
		std::cout << in[i] << " ";
	}
	std::cout << std::endl;

	std::cout << std::endl << "--Forward  W: " << W.size()<< std::endl;
	for (size_t i = 0; i < W.size(); i++) {
		std::cout << W[i] << " ";
	}
	std::cout << std::endl;

    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (params.has_bias_) {
        out[i] += bias[i];
      }
    }

	std::cout << std::endl<<"--Forward out: "<< out.size()<<std::endl;
	for (size_t i = 0; i < out.size(); i++) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;

  });

  

}

inline void fully_connected_op_internal(const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
  for (size_t sample = 0; sample < prev_out.size(); sample++) {


    //sum(GRAD * W1 + GRAD2*W2+...)
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

	std::cout << std::endl << "-- Forward: Sum of all forwarding gradient * weight : " << prev_delta[sample].size() << std::endl;
	for (size_t i = 0; i < prev_delta[sample].size(); i++) {
		std::cout << prev_delta[sample][i] << " ";
	}
	std::cout << std::endl;


    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                          r.end() - r.begin(),
                          &dW[sample][c * params.out_size_ + r.begin()]);
      }

      if (params.has_bias_) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] += curr_delta[sample][i];
        }
      }
    });
  }

  std::cout << std::endl << "--Weights update dW: " << dW[0].size() << std::endl;
  for (size_t i = 0; i < dW[0].size(); i++) {
	  std::cout << dW[0][i] << " ";
  }
  std::cout << std::endl;
}

}  // namespace kernels
}  // namespace tiny_dnn
