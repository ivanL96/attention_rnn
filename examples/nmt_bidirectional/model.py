import numpy as np
from tensorflow.python.keras.layers import Input, GRU, LSTM, Dense, Concatenate, TimeDistributed, Bidirectional, CuDNNLSTM, CuDNNGRU
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from layers.attention import AttentionLayer

IS_BIDIRECTIONAL = True
RECURRENT = GRU
# RECURRENT = CuDNNLSTM

# USED ONLY IF THE CuDNN IS ENABLED
RECURRENT_DROPOUT = .1 

LR = 0.01
HIDDEN_DIM = 50

def create_rnn_layer(size, bi_layer=IS_BIDIRECTIONAL, name=''):
    if RECURRENT in [CuDNNGRU, CuDNNLSTM]:
        layer = RECURRENT(size, return_sequences=True, return_state=True, name=name)
    else:
        layer = RECURRENT(size, return_sequences=True, return_state=True, recurrent_dropout=RECURRENT_DROPOUT, name=name)
    if bi_layer:
        layer = Bidirectional(layer)
    return layer


def get_state(recurrent_layer, bi_layer=IS_BIDIRECTIONAL, is_tensors=True):
    def concat(arrs, is_tensors=is_tensors):
        if [el for el in arrs if el is not None]:
            return Concatenate(axis=-1)(arrs) if is_tensors else np.concatenate(arrs, axis=-1)
    
    def get_state_el(states, idx): # safe state fetching
        try:
            return states[idx]
        except:
            return None

    # print('recurrent_layer: ', recurrent_layer)
    out = recurrent_layer[0]
    initial_states = recurrent_layer[1:]
    if RECURRENT in [CuDNNGRU, GRU]:
        if bi_layer:  # BIDIR returns - (out, [fwd_state, back_state]) ELSE (out, state)
            # print('initial_states???? ', initial_states)
            # if len(initial_states) >= 2:
            fwd_state = get_state_el(initial_states, 0)
            back_state = get_state_el(initial_states, 1)
            return out, concat([fwd_state, back_state])
        else:
            return out, initial_states[0]
    elif RECURRENT in [CuDNNLSTM, LSTM]:
        if bi_layer:
            fwd_states = Concatenate(axis=0)(initial_states[0:2])
            back_states = Concatenate(axis=0)(initial_states[2:])
            print('initial_states lstm???? ', [fwd_states.shape, back_states.shape])
            return out, concat([fwd_states, back_states])
        else:
            h = initial_states[0]
            c = initial_states[1]
            return out, concat([h, c])


def define_nmt(batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
    """ Defining a NMT model """

    HIDDEN_SIZE_DEC = HIDDEN_DIM*2 #if IS_BIDIRECTIONAL else HIDDEN_DIM
    
    # Define an input sequence and process it.
    if batch_size:
        encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')
    else:
        encoder_inputs = Input(shape=(en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(shape=(fr_timesteps - 1, fr_vsize), name='decoder_inputs')

    # Encoder GRU
    encoder_gru = create_rnn_layer(HIDDEN_DIM, name='encoder_rnn')
    encoder_out, encoder_states = get_state(encoder_gru(encoder_inputs))

    # Set up the decoder GRU, using `encoder_states` as initial state.
    decoder_gru = create_rnn_layer(HIDDEN_SIZE_DEC, bi_layer=False, name='decoder_rnn')
    # encoder_states = Concatenate(axis=-1)(encoder_states)
    print('encoder_states!!!!: ', encoder_states)
    d = decoder_gru(decoder_inputs, initial_state=encoder_states)
    decoder_out, decoder_state = get_state(d, bi_layer=False)

    # encoder_gru = Bidirectional(RECURRENT(HIDDEN_DIM, return_sequences=True, return_state=True, name='encoder_gru'))
    # encoder_out, fwd, back = encoder_gru(encoder_inputs)

    # decoder_gru = RECURRENT(HIDDEN_SIZE_DEC, return_sequences=True, return_state=True, name='decoder_gru')
    # encoder_states = Concatenate(axis=-1)([fwd, back])
    # print('encoder_states!!!!: ', encoder_states)
    # decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_states)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer=RMSprop(lr=LR), loss='categorical_crossentropy')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
    
    #v1
    encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state = encoder_gru(encoder_inf_inputs)
    print('V1 INFER ENCODER: ', encoder_inf_out.shape, encoder_inf_fwd_state.shape, encoder_inf_back_state.shape)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state])
    #v2
    # encoder_inf_out, encoder_inf_states = get_state(encoder_gru(encoder_inf_inputs))
    # print('V2 INFER ENCODER: ', encoder_inf_out.shape, encoder_inf_states.shape)
    # encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, ] + encoder_inf_states)

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, HIDDEN_SIZE_DEC), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, HIDDEN_SIZE_DEC), name='decoder_init')

    decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

    return full_model, encoder_model, decoder_model

