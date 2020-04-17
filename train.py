import tensorflow as tf
from random_gene import random_input_generator, input_helper
from tensorflow.keras.layers import Lambda as Lambda
import numpy as np
import json
# some tokens
PAD = 0 
SOS = 1 # start of sentence
EOS = 2 # end of sentence

vocab_amount = 2000
batch_size = 512

# word embedding vector size
input_embedding_size = 300 

# LSTM (1 layer in the model)
encoder_hidden_units = 200
decoder_hidden_units = encoder_hidden_units*2



###############################
#######    training     #######
###############################

# Define an input sequence and process it.
encoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name='encoder_input')
decoder_inputs = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name='decoder_input')

# embedding vectors.
embedding_matrix = tf.keras.layers.Embedding(vocab_amount, input_embedding_size)

# bi-LSTM encoder
_LSTM_encoder = tf.keras.layers.LSTM(encoder_hidden_units, return_state=True, return_sequences=True)
LSTM_encoder =  tf.keras.layers.Bidirectional(_LSTM_encoder,merge_mode='concat') 

# LSTM decoder
LSTM_decoder = tf.keras.layers.LSTM(decoder_hidden_units, return_state=True, return_sequences=True)

# after LSTM decoder, add a dense layer.
decoder_dense =  tf.keras.layers.Dense(vocab_amount, activation='softmax')
# okay, let's build the model!
# embed the input. ( one-hot to meaningful, embedding vector)
emb_encoder_input = embedding_matrix(encoder_inputs)
emb_decoder_input = embedding_matrix(decoder_inputs)

# into LSTM encoder
enco_output, fw_h, fw_c, bw_h, bw_c = LSTM_encoder(emb_encoder_input)

state_h = tf.keras.layers.concatenate([fw_h, bw_h], axis=1)
state_c = tf.keras.layers.concatenate([fw_c, bw_c], axis=1)
encoder_states = [state_h, state_c]

# into LSTM decoder
deco_output, _, _ = LSTM_decoder(emb_decoder_input, initial_state=encoder_states)
# dense layer (with softmax as activatiom)
decoder_outputs = decoder_dense(deco_output)
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
adam = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')
###############################
#######    inference    #######
###############################

encoder_model = tf.keras.Model(encoder_inputs, [enco_output, encoder_states])


decoder_state_h = tf.keras.layers.Input(shape=(decoder_hidden_units,), dtype=tf.float32)
decoder_state_c = tf.keras.layers.Input(shape=(decoder_hidden_units,), dtype=tf.float32)
decoder_states = [decoder_state_h, decoder_state_c]

inf_deco_output, inf_h, inf_c = LSTM_decoder(emb_decoder_input, initial_state=decoder_states)

inf_final_output = decoder_dense(inf_deco_output)

decoder_model = tf.keras.Model([decoder_inputs, decoder_state_h, decoder_state_c],
                               [inf_final_output, inf_h, inf_c])

def decode_sequence(input_seq, max_decoder_seq_length):
    encoder_model_output = encoder_model.predict(input_seq)

    input_seq_output = encoder_model_output[0]
    states_value = encoder_model_output[1:3]

    target_seq = np.array([[SOS]])
    total_loop = 0
    stop_condition = False
    output_sentence = []

    while not stop_condition:
        output_tokens, L1_h, L1_c = decoder_model.predict([target_seq]+ states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        output_sentence.append(sampled_token_index)

        if (sampled_token_index == EOS or total_loop >  max_decoder_seq_length):
           stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.array([[sampled_token_index]])
        # Update states
        states_value = [L1_h, L1_c]

        total_loop += 1

    return output_sentence

with open('train_fine.json','r',encoding='utf-8') as file1:
    dict_train = json.load(file1)
with open('test_fine.json','r',encoding='utf-8') as file1:
    dict_test = json.load(file1)

raw_list=[]
raw_target=[]
for key,value in dict_train.items():
    list1=[value['liked_count']+3,value['comment_count']+3]
    for idx,i in enumerate(list1):
        if i>1000:
            list1[idx]=997
    raw_list.append(list1)
    if value['like_count']>=1000:
        raw_target.append('997')
    else:
        raw_target.append(value['like_count'])

for epoch in range(80):
    # to generate random input
    
    input_data, input_length = random_input_generator(raw_list)
    encoder_input_data, decoder_input_data, decoder_target_data  = input_helper(input_data,
                                                                                raw_target,
                                                                                np.max(input_length),
                                                                                start_token=SOS,
                                                                                end_token=EOS,
                                                                                pad_token=PAD)
    print(encoder_input_data.shape) 
    print(decoder_input_data.shape)
    print(decoder_target_data.shape)
    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              batch_size=batch_size,
              epochs=1,
              validation_split=0.3)
    model.save('try.h5')
    
    for _ in range(2):
        
        prediction = decode_sequence(np.reshape(encoder_input_data[epoch+_], (1, -1)),
                                     max_decoder_seq_length=2)
        print("         input:", encoder_input_data[epoch+_])
        print("        target:", decoder_target_data[epoch+_])
        print("    prediction:", prediction)

