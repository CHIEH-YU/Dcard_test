import numpy as np
import random
import json


def random_input_generator(raw_list):
    random_list = []
    random_length = []
    for i in range(len(raw_list)):
        #this_array=np.random.randint(100,size=4)
        this_array=np.array(raw_list[i])
        random_list.append(this_array)
        random_length.append(2)
    return random_list, np.array(random_length)

def input_helper(random_list,target_list,length_upper_limit, start_token, end_token, pad_token):
    length_upper_limit = length_upper_limit + 2 # to include start/end tokens
    length_upper_limit = 2 + 2    
    encoder_input = np.zeros((len(random_list),length_upper_limit)).astype(np.int32)
    decoder_input = np.zeros((len(random_list),length_upper_limit)).astype(np.int32)
    decoder_target = np.zeros((len(random_list),length_upper_limit)).astype(np.int32)
    
    for idx, this_data in enumerate(random_list):
        this_length = len(this_data)
        this_encoder_input_pads = np.array([pad_token] * (length_upper_limit - (this_length + 1))).astype(np.int32)
        this_decoder_input_pads = np.array([pad_token] * (length_upper_limit - (this_length + 2))).astype(np.int32)
        this_decoder_target_pads = np.array([pad_token] * (length_upper_limit - (this_length + 1))).astype(np.int32)
 
        this_encoder_input = np.concatenate((this_data.copy(),
                                             np.array([end_token]),
                                             this_encoder_input_pads))
        this_decoder_input = np.concatenate((np.array([start_token]), 
                                             this_data.copy(),
                                             np.array([end_token]),
                                             this_decoder_input_pads))
        this_data[-1]=target_list[idx]
        this_decoder_target = np.concatenate((this_data.copy(),
                                              np.array([end_token]),
                                              this_decoder_target_pads))
        encoder_input[idx] = this_encoder_input
        decoder_input[idx] = this_decoder_input
        decoder_target[idx] = this_decoder_target

    return encoder_input, decoder_input, decoder_target

