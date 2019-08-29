import re
import random

def load_data():
    dis = []
    lengths = []

    with open('C:\Projects\dialogues.txt', 'r', encoding='utf-8') as f:
        QA = f.read().split('\n\n')
        random.shuffle(QA)
        
        for dialogue in QA:        
            dialogue_elems = dialogue.split('\n') 
            
            if len(dialogue_elems) > 2:
                for i,a in enumerate(dialogue_elems):
                    if i+1 < len(dialogue_elems):            
                        dis.append('\n'.join(dialogue_elems[i:i+2]))
            else:
                if dialogue != '':
                    dis.append(dialogue)

    print('dis:   ', dis[:5])
    print('Dialogues length: ', len(dis))
    return dis

pattern = re.compile('([^\s\w]|_)+')
def clean_text(text):    
    text = text.lower()
    text = text.replace('╦', 'т').replace('╕','е').replace('\t', '')
    text = pattern.sub('', text)

    if len(text) > 1 and text[:2] == '- ':
        text = text[2:]
    return text

def split_to_qa(dis):
    input_data = []
    output_data = []
    NB_SAMPLES = 10000#len(dis)
    vocab = []

    chars = ['\t', '\n']

    MAX_ENC_LEN = 0
    MAX_DEC_LEN = 0

    for el in dis[:NB_SAMPLES]:
        if len(el.split('\n')) == 2:
            enc = clean_text(el.split('\n')[0])
            dec = clean_text(el.split('\n')[1])
            
            input_data.append(enc)
            output_data.append(dec)
            
            if MAX_ENC_LEN < len(enc):
                MAX_ENC_LEN = len(enc)
            if MAX_DEC_LEN < len(dec):
                MAX_DEC_LEN = len(dec)
            
            chars += [ch for ch in dec+enc if ch not in chars]
            
            vocab += enc.split() + dec.split()

    vocab = set(vocab)
    chars = sorted(set(chars))

    print(len(input_data), input_data[:3])
    print(len(output_data), output_data[:3])
    print(len(vocab))
    return input_data, output_data
