import keras
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import Callback
import random as randint

with open('miracles.txt', 'r') as file:
    lyrics = file.read()

chars = list(set(lyrics))
data_size, vocab_size = len(lyrics), len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

sentence_length = 50
sentences = []
next_chars = []

for i in range(data_size - sentence_length):
    sentences.append(lyrics[i: i + sentence_length])
    next_chars.append(lyrics[i+sentence_length])

num_sentence = len(sentences)

x = np.zeros((num_sentence, sentence_length, vocab_size))
y = np.zeros((num_sentence, vocab_size), dtype=np.bool)

for i, sentences in enumerate(sentences):
    for t, char in enumerate(sentences):
        x[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(256, input_shape=(sentence_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

def sample_from_model(model, sample_length=100):
    seed = randint(0, data_size - sentence_length)
    seed_sentence = lyrics[seed: seed + sentence_length]

    x_pred = np.zeros((1, sentence_length, vocab_size), dtype=np.bool)

    for t, char in enumerate(sentences):
        x_pred[0, t, char_to_idx[char]] = 1

    generated_text = ''

    for i in range(sample_length):
        prediction = np.argmax(model.predict(x_pred))
        generated_text += idx_to_char[prediction]
        activations = np.zeros((1, sentence_length, vocab_size), dtype=np.bool)
        activations[0, 0, prediction] = 1
        x_pred = np.concatenate((x_pred[:, 1:, :], activations), axis=1)

    return generated_text

class SamplerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        generated_text = sample_from_model(self.model)
        print("\nGenerated text")
        print('-' * 32)
        print(generated_text)


sampler_callback = SamplerCallback()
model.fit(x, y, epochs=30, batch_size=256, callbacks=[sampler_callback])
generated_text = sample_from_model(model, sample_length=1000)
print("\nGenerated text")
print('-' * 32)
print(generated_text)


