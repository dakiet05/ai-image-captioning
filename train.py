import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from preprocess import load_tokenizer

def train_model(model, train_features, train_sequences, test_features, test_sequences, epochs, tokenizer):
    X1_train, X2_train, y_train = [], [], []
    for i in range(len(train_sequences)):
        X1_train.append(train_features[i // 5])
        X2_train.append(train_sequences[i][:-1])
        y_train.append(train_sequences[i][1:])
    
    X1_train = np.array(X1_train)
    X2_train = np.array(X2_train)
    y_train = np.array(y_train)
    
    print(f"Shape of X2 after padding: {X2_train.shape}")
    print(f"Shape of y after padding: {y_train.shape}")
    print(f"Shape of expanded_features: {X1_train.shape}")
    
    X1_test, X2_test, y_test = [], [], []
    for i in range(len(test_sequences)):
        X1_test.append(test_features[i // 5])
        X2_test.append(test_sequences[i][:-1])
        y_test.append(test_sequences[i][1:])
    
    X1_test = np.array(X1_test)
    X2_test = np.array(X2_test)
    y_test = np.array(y_test)
    
    print(f"Shape of X2 after padding: {X2_test.shape}")
    print(f"Shape of y after padding: {y_test.shape}")
    print(f"Shape of expanded_features: {X1_test.shape}")
    
    # Learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_test, X2_test], y_test),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )
    
    # Lưu mô hình
    model.save('model.h5')
    
    # Đánh giá BLEU score với beam search
    bleu_score = evaluate_bleu(model, X1_test, X2_test, y_test, tokenizer, beam_width=5)
    print(f"BLEU-1 score on test set: {bleu_score}")
    
    return history

def evaluate_bleu(model, X1_test, X2_test, y_test, tokenizer, beam_width=5):
    predictions = []
    references = []
    
    # Tạo từ điển ánh xạ từ index về từ
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    
    for i in range(len(X1_test)):
        input_feature = np.expand_dims(X1_test[i], axis=0)
        y_true = y_test[i]
        
        # Beam search
        sequences = [[[], 0.0]]  # [[sequence, score], ...]
        max_length = X2_test.shape[1] + 1  # max_length = 39 (X2_test là 38, cộng 1 để bao gồm token cuối)
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                if len(seq) > 0 and seq[-1] == 0:  # Nếu đã gặp padding, dừng
                    all_candidates.append([seq, score])
                    continue
                input_seq = np.zeros((1, max_length-1), dtype=np.int32)  # Sửa shape thành max_length-1 (38)
                input_seq[0, :len(seq)] = seq
                probs = model.predict([input_feature, input_seq], verbose=0)[0, len(seq)]
                top_k = np.argsort(probs)[-beam_width:]  # Lấy top-k indices
                for next_word in top_k:
                    new_seq = seq + [next_word]
                    new_score = score + np.log(probs[next_word] + 1e-10)
                    all_candidates.append([new_seq, new_score])
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Lấy sequence có score cao nhất
        predicted_seq = sequences[0][0]
        
        # Chuyển từ index về từ
        reference = [index_to_word.get(idx, '') for idx in y_true if idx != 0]
        prediction = [index_to_word.get(idx, '') for idx in predicted_seq if idx != 0]
        
        references.append([reference])
        predictions.append(prediction)
    
    bleu_score = corpus_bleu(references, predictions, weights=(1, 0, 0, 0))
    return bleu_score