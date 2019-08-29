import tensorflow.keras as keras

from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style
import os, sys
import time

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
if project_path not in sys.path:
    sys.path.append(project_path)

from examples.utils.data_helper import read_data, sents2sequences
from examples.nmt_bidirectional.model import define_nmt, get_state
from examples.utils.model_helper import plot_attention_weights
from examples.utils.logger import get_logger

from preprocess import load_data, split_to_qa

np.random.seed(100)

base_dir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
logger = get_logger("examples.nmt_bidirectional.train", os.path.join(base_dir, 'logs'))

BATCH_SIZE = 180
en_timesteps, fr_timesteps = 10, 10
TRAIN_TEST_SPLIT = 0.20
N_EPOCHS = 1


def get_data():

    """ Getting randomly shuffled training / testing data """
    # en_text = read_data(os.path.join(project_path, 'data', 'small_vocab_en.txt'))
    # fr_text = read_data(os.path.join(project_path, 'data', 'small_vocab_fr.txt'))
    en_text, fr_text = split_to_qa(load_data())
    logger.info('Length of text: {}'.format(len(en_text)))

    fr_text = ['sos ' + sent[:-1] + 'eos' if sent.endswith('.') else 'sos ' + sent + ' eos' for sent in fr_text]

    # inds = np.arange(len(en_text))
    # np.random.shuffle(inds)

    # train-test split
    tr_en_text, ts_en_text, tr_fr_text, ts_fr_text = train_test_split(en_text, fr_text, test_size=TRAIN_TEST_SPLIT, shuffle=False)
    # train_inds = inds[:train_size]
    # test_inds = inds[train_size:]
    # tr_en_text = [en_text[ti] for ti in train_inds]
    # tr_fr_text = [fr_text[ti] for ti in train_inds]

    # ts_en_text = [en_text[ti] for ti in test_inds]
    # ts_fr_text = [fr_text[ti] for ti in test_inds]

    logger.info("Average length of an Encoder sentence: {}".format(
        np.mean([len(en_sent.split(" ")) for en_sent in tr_en_text])))
    logger.info("Average length of a Decoder sentence: {}".format(
        np.mean([len(fr_sent.split(" ")) for fr_sent in tr_fr_text])))
    return tr_en_text, tr_fr_text, ts_en_text, ts_fr_text


def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    fr_seq = sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    logger.info('Vocabulary size (Encoder): {}'.format(np.max(en_seq)+1))
    logger.info('Vocabulary size (Decoder): {}'.format(np.max(fr_seq)+1))
    logger.debug('Encoder text shape: {}'.format(en_seq.shape))
    logger.debug('Decoder text shape: {}'.format(fr_seq.shape))
    return en_seq, fr_seq


def train(full_model, infer_enc_model, infer_dec_model, en_seq, fr_seq):
    """ Training the model """

    for ep in range(N_EPOCHS):
        losses = []
        start = time.time()
        for bi in tqdm(range(0, en_seq.shape[0] - BATCH_SIZE, BATCH_SIZE)):

            en_onehot_seq = to_categorical(en_seq[bi:bi + BATCH_SIZE, :], num_classes=en_vsize)
            fr_onehot_seq = to_categorical(fr_seq[bi:bi + BATCH_SIZE, :], num_classes=fr_vsize)

            full_model.train_on_batch([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
                                    batch_size=BATCH_SIZE, verbose=0)

            losses.append(l)
        end = time.time()
        # if (ep + 1) % 5 == 0: 
            # save model every 5 epochs
            # save_model(full_model, ep+1)

        # show test results after epoch
        test_inferring(infer_enc_model, infer_dec_model)
        logger.info("Elapsed: {} sec. Loss in epoch {}/{}: {}".format(round(end-start, 3), ep + 1, N_EPOCHS, np.mean(losses)))


def infer_nmt(encoder_model, decoder_model, test_en_seq, en_vsize, fr_vsize):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param fr_vsize: int
    :return:
    """

    test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], fr_vsize)
    test_en_onehot_seq = to_categorical(test_en_seq, num_classes=en_vsize)
    test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

    enc_outs, enc_states = get_state(encoder_model.predict(test_en_onehot_seq), is_tensors=False)
    dec_state = enc_states
    print('WHAT ARE YOU 0 : ', enc_outs.shape, dec_state.shape) 

    enc_outs, fwd, back = encoder_model.predict(test_en_onehot_seq)
    dec_state = np.concatenate([fwd, back], axis=-1)
    print('WHAT ARE YOU 1 : ', enc_outs.shape, dec_state.shape)
    
    attention_weights = []
    fr_text = [] # list of generated words

    for i in range(fr_timesteps):

        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, test_fr_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break
        test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
        test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

        attention_weights.append((dec_ind, attention)) # for further visualization
        if not fr_index2word[dec_ind] == 'eos':
            fr_text.append(fr_index2word[dec_ind])

    return ' '.join(fr_text), attention_weights


def save_model(model, fname=''):
    if not os.path.exists(os.path.join('..', 'h5.models')):
        os.mkdir(os.path.join('..', 'h5.models'))
    model.save(os.path.join('..', 'h5.models', 'nmt_{}.h5'.format(fname)))


def test_inferring(infer_enc_model, infer_dec_model, plot=False):
    """ Inferring with trained model """
    rand_test_ids = np.random.randint(0, len(ts_en_text), size=10)
    for rid in rand_test_ids:
        test_en = ts_en_text[rid]
        # logger.info('\tRequest: {}'.format(test_en))
        print('Request: {}'.format(test_en))

        test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)
        test_fr, attn_weights = infer_nmt(encoder_model=infer_enc_model, decoder_model=infer_dec_model,
            test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize)
        print(Fore.GREEN + 'Response: {}'.format(test_fr) + Style.RESET_ALL)
        # print()

        if plot:
            """ Attention plotting """
            plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word,
                                base_dir=base_dir, filename='attention_{}.png'.format(rid))


if __name__ == '__main__':
    tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data()

    """ Defining tokenizers """
    en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    en_tokenizer.fit_on_texts(tr_en_text)

    fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    fr_tokenizer.fit_on_texts(tr_fr_text)

    """ Index2word """
    en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
    fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

    """ Getting preprocessed data """
    en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)

    en_vsize = max(en_tokenizer.index_word.keys()) + 1
    fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

    """ Defining the full model """
    full_model, infer_enc_model, infer_dec_model = define_nmt(batch_size=BATCH_SIZE,
        en_timesteps=en_timesteps, fr_timesteps=fr_timesteps,
        en_vsize=en_vsize, fr_vsize=fr_vsize)

    train(full_model, 
        infer_enc_model, infer_dec_model, 
        en_seq, fr_seq)

    """ Save model """
    save_model(full_model)

    test_inferring(infer_enc_model, infer_dec_model, plot=False)
