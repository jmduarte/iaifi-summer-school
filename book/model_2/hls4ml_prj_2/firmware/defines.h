#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_5 32
#define N_LAYER_8 32
#define N_LAYER_11 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<10,4> model_default_t;
typedef ap_fixed<10,4> input_t;
typedef ap_fixed<10,4> layer2_t;
typedef ap_fixed<12,6> fc1_weight_t;
typedef ap_fixed<10,4> fc1_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<10,4> relu1_default_t;
typedef ap_fixed<18,8> relu1table_t;
typedef ap_fixed<10,4> layer4_t;
typedef ap_fixed<10,4> layer5_t;
typedef ap_fixed<10,4> fc2_weight_t;
typedef ap_fixed<10,4> fc2_bias_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<10,4> relu2_default_t;
typedef ap_fixed<18,8> relu2table_t;
typedef ap_fixed<10,4> layer7_t;
typedef ap_fixed<10,4> layer8_t;
typedef ap_fixed<10,4> fc3_weight_t;
typedef ap_fixed<10,4> fc3_bias_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<10,4> relu3_default_t;
typedef ap_fixed<18,8> relu3table_t;
typedef ap_fixed<10,4> layer10_t;
typedef ap_fixed<10,4> layer11_t;
typedef ap_fixed<10,4> output_weight_t;
typedef ap_fixed<10,4> output_bias_t;
typedef ap_uint<1> layer11_index;
typedef ap_fixed<10,4> softmax_default_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> softmaxexp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> softmaxinv_table_t;
typedef ap_fixed<10,4> result_t;
typedef ap_fixed<18,8> softmax_table_t;

#endif
