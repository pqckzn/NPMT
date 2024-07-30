import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
#Load data
import os

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r",encoding='utf-8') as f:
        data = f.read()

    return data.split('\n')
english_sentences = load_data('en_sents.txt')
viet_sentences = load_data('vi_sents.txt')
print('Dataset Loaded')
# Preprocess
def tokenize(x):
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

# Padding
def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

# Pre-process Pipeline
def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
# Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_viet_sentences, english_tokenizer, viet_tokenizer =\
    preprocess(english_sentences, viet_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_viet_sequence_length = preproc_viet_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
viet_vocab_size = len(viet_tokenizer.word_index)
print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max viet sentence length:", max_viet_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("viet vocabulary size:", viet_vocab_size)
# Ids Back to Text
def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')
# Model 
from tensorflow.keras.models import Sequential
def model_final(input_shape, output_sequence_length, english_vocab_size, viet_vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(viet_vocab_size,activation='softmax')))
    learning_rate = 0.005
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model
""" tests.test_model_final(model_final) """
print('Final Model Loaded')

# Prediction
def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed viet data
    :param x_tk: English tokenizer
    :param y_tk: viet tokenizer
    """
    # TODO: Train neural network using model_final
    tmp_X = pad(preproc_english_sentences)
    tmp_Y = pad(preproc_viet_sentences)
    model = model_final(tmp_X.shape,
                        tmp_Y.shape[1],
                        len(english_tokenizer.word_index)+1,
                        len(viet_tokenizer.word_index)+1)
    
    model.fit(tmp_X, tmp_Y, batch_size = 256, epochs = 10, validation_split = 0.2)
 
    ## DON'T EDIT ANYTHING BELOW THIS LINE
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'that is what I want to know'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('đó là những gì tôi muốn biết')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


final_predictions(preproc_english_sentences, preproc_viet_sentences, english_tokenizer, viet_tokenizer)