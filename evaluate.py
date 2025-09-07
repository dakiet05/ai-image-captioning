import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def beam_search_predict(model, feature, tokenizer, max_length, beam_width=5):
    start_token = tokenizer.word_index['startseq']
    end_token = tokenizer.word_index['endseq']

    sequences = [[start_token]]
    scores = [0.0]
    feature_expanded = np.expand_dims(feature, axis=0)

    for step in range(max_length - 1):
        candidates = []
        for i, seq in enumerate(sequences):
            score = scores[i]
            seq_padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_length-1, padding='post')
            pred = model.predict([feature_expanded, seq_padded], verbose=0)
            timestep = min(step, pred.shape[1] - 1)
            probs = pred[0, timestep, :]
            top_words = np.argsort(probs)[-beam_width:]
            for w in top_words:
                new_seq = seq + [w]
                new_score = score + float(np.log(probs[w] + 1e-10))
                candidates.append((new_seq, new_score))
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        sequences = [s for s, _ in ordered]
        scores = [sc for _, sc in ordered]
        if any(s[-1] == end_token for s in sequences):
            break

    best_seq = sequences[0]
    inv = tokenizer.index_word
    words = [inv.get(w, '') for w in best_seq if w in inv and w not in (start_token, end_token)]
    return ' '.join([w for w in words if w]).strip()

def evaluate_model(model, test_features, test_sequences, tokenizer, max_length):
    refs, preds = [], []
    smooth = SmoothingFunction().method4
    for i in range(len(test_features)):
        feat = test_features[i]
        true_seq = test_sequences[i]
        inv = tokenizer.index_word
        true_text = ' '.join([inv.get(w, '') for w in true_seq if w != 0]).strip()
        refs.append(true_text.split())

        pred_caption = beam_search_predict(model, feat, tokenizer, max_length)
        preds.append(pred_caption.split())

        print(f"True: {true_text}")
        print(f"Pred: {pred_caption}")
        bleu1 = sentence_bleu([refs[-1]], preds[-1], weights=(1, 0, 0, 0), smoothing_function=smooth)
        print(f"BLEU-1: {bleu1:.4f}")

    bleu_scores = [sentence_bleu([r], p, weights=(1, 0, 0, 0), smoothing_function=smooth) for r, p in zip(refs, preds)]
    print(f"Average BLEU-1: {np.mean(bleu_scores):.4f}")
