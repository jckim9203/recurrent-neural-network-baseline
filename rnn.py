import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = """
이번 주말에 영화 보고 싶은데, 우리 저녁 먹고 영화보러 갈까?
저녁 먹고 영화보니까 좋다.
오늘은 날씨가 좋아서 공원에 가고 싶어.
새로운 맛집이 생겼다고 하니 가볼까?
일요일에는 가족과 함께 시간을 보내고 싶어.
저녁에 운동하고 책 읽을까 생각 중이야.
저녁 먹고 산책하는게 내 일상이야.
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
tokenizer.word_index

input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
input_sequences

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
input_sequences

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(y)
y

rnn_model = Sequential([
    Embedding(total_words, 10, input_length=max_sequence_len-1),
    SimpleRNN(50),
    Dense(total_words, activation='softmax')
])

rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X, y, epochs=100)

lstm_model = Sequential([
    Embedding(total_words, 10, input_length=max_sequence_len-1),
    LSTM(50),
    Dense(total_words, activation='softmax')
])

lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X, y, epochs=100)

def predict_next_word(model, tokenizer, text_input, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word = tokenizer.index_word[np.argmax(predicted)]
    return predicted_word

seed_text = "오늘 저녁먹고 공원"
print("RNN: ", predict_next_word(rnn_model, tokenizer, seed_text, max_sequence_len))
print("LSTM: ", predict_next_word(lstm_model, tokenizer, seed_text, max_sequence_len))
