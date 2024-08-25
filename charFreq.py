from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

def get_ngrams(text, n):
    return [''.join(gram) for gram in zip(*[text[i:] for i in range(n)])]

def calculate_normalized_frequencies(texts_by_file_type, n=1):
    ngram_frequencies = {}
    normalized_frequencies = {}
    unique_ngrams = set()
    
    train_data = texts_by_file_type['train']
    
    for language, data in train_data.items():
        full_text = ''.join(data)
        ngrams = get_ngrams(full_text, n)
        unique_ngrams.update(ngrams)
        ngram_counter = Counter(ngrams)
        
        total_ngrams = sum(ngram_counter.values())
        normalized_freq = {ngram: count / total_ngrams for ngram, count in ngram_counter.items()}
        
        ngram_frequencies[language] = ngram_counter
        normalized_frequencies[language] = normalized_freq

    ngram_to_index = {ngram: i for i, ngram in enumerate(unique_ngrams)}
    
    normalized_vectors = {}
    for language, freq in normalized_frequencies.items():
        vec = np.zeros(len(unique_ngrams))
        for ngram, frequency in freq.items():
            vec[ngram_to_index[ngram]] = frequency
        normalized_vectors[language] = sparse.csr_matrix(vec)
    
    return ngram_frequencies, normalized_vectors, ngram_to_index

def calculate_similarity(vec1, vec2, metric):
    if metric == 'cosine':
        return vec1.dot(vec2.T).toarray()[0, 0] / (np.sqrt(vec1.power(2).sum()) * np.sqrt(vec2.power(2).sum()))
    elif metric == 'euclidean':
        return -np.sqrt((vec1 - vec2).power(2).sum())
    elif metric == 'max_inner_product':
        return vec1.dot(vec2.T).toarray()[0, 0]
    elif metric == 'manhattan':
        return -(vec1 - vec2).abs().sum()
    elif metric == 'jaccard':
        intersection = np.minimum(vec1, vec2).sum()
        union = np.maximum(vec1, vec2).sum()
        return intersection / union if union != 0 else 0
    elif metric in ['kl_divergence', 'jensen_shannon']:
        vec1 = vec1.toarray().flatten() + 1e-10
        vec2 = vec2.toarray().flatten() + 1e-10
        if metric == 'kl_divergence':
            return -np.sum(vec1 * np.log(vec1 / vec2))
        else:
            m = 0.5 * (vec1 + vec2)
            return -0.5 * (np.sum(vec1 * np.log(vec1 / m)) + np.sum(vec2 * np.log(vec2 / m)))

def predict_sentence_language(sentence, train_normalized_vecs, ngram_to_index, similarity_metric='cosine', n=1):
    ngrams = get_ngrams(sentence, n)
    ngram_counter = Counter(ngrams)
    total_ngrams = sum(ngram_counter.values())
    
    vec = np.zeros(len(ngram_to_index))
    for ngram, count in ngram_counter.items():
        if ngram in ngram_to_index:
            vec[ngram_to_index[ngram]] = count / total_ngrams
    sentence_vec = sparse.csr_matrix(vec)
    
    similarities = {lang: calculate_similarity(sentence_vec, train_vec, similarity_metric) 
                    for lang, train_vec in train_normalized_vecs.items()}
            
    return max(similarities, key=similarities.get)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", fontsize=5,
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def evaluate_language_prediction(test_data, train_normalized_vecs, ngram_to_index, similarity_metric='cosine', n=1, n_words=10):
    predictions = []
    actuals = []
    correct_sample_found = False
    misclassified_sample_found = False

    for language, sentences in test_data.items():
        print(language)
        fixed_sentences = create_fixed_word_sentences(sentences, n_words)
        for sentence in fixed_sentences:
            predicted_language = predict_sentence_language(sentence, train_normalized_vecs, ngram_to_index, similarity_metric, n)
            predictions.append(predicted_language)
            actuals.append(language)
            
            if language == predicted_language and not correct_sample_found:
                print("\nCorrect Sample:")
                print(f"Text: {sentence}")
                print(f"True Language: {language}")
                print(f"Predicted Language: {predicted_language}")
                correct_sample_found = True
            
            if language != predicted_language and not misclassified_sample_found:
                print("\nMisclassified Sample:")
                print(f"Text: {sentence}")
                print(f"True Language: {language}")
                print(f"Predicted Language: {predicted_language}")
                misclassified_sample_found = True
        
        correct_sample_found = False
        misclassified_sample_found = False

    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')
    cm = confusion_matrix(actuals, predictions)
    labels = sorted(set(actuals + predictions))

    return accuracy, f1, cm, labels

def run_evaluation(texts_by_file_type, n_gram=1, similarity_metric='cosine', n_words=100):
    ngram_frequencies, normalized_vectors, ngram_to_index = calculate_normalized_frequencies(texts_by_file_type, n=n_gram)
    test_data = texts_by_file_type['test']
    
    accuracy, f1, cm, labels = evaluate_language_prediction(
        test_data, normalized_vectors, ngram_to_index, similarity_metric=similarity_metric, n=n_gram, n_words=n_words
    )
    
    print(f"\nN-gram: {n_gram}, Metric: {similarity_metric}, Word count: {n_words}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    save_confusion_matrix(cm, labels, similarity_metric, n_words, f1, accuracy)
    plot_confusion_matrix(cm, classes=labels, title=f'Confusion Matrix - N-gram: {n_gram}, Metric: {similarity_metric}')
    plt.show()

    return accuracy, f1, cm

# Implementing save_confusion_matrix function
def save_confusion_matrix(cm, classes, metric, n_words, f1, accuracy):
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{metric} - #n words {n_words} - F1: {f1:.4f} - Acc: {accuracy:.4f}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    # Construct the filename with relevant metrics and configurations
    filename = f"experiments/confusion_{metric}_nwords{n_words}_f1_{f1:.4f}_acc_{accuracy:.4f}.png"
    plt.savefig(filename)
    plt.close() 

def run_all_tests(texts_by_file_type, n_grams, similarity_metrics, n_words_list):
    results = {}

    for n in n_grams:
        for metric in similarity_metrics:
            for n_words in n_words_list:
                print(f"Running evaluation for n-gram={n}, metric={metric}, n_words={n_words}")
                accuracy, f1, cm = run_evaluation(texts_by_file_type, n_gram=n, similarity_metric=metric, n_words=n_words)
                results[(n, metric, n_words)] = (accuracy, f1)
                log_results(n, metric, n_words, accuracy, f1)


    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    ind = np.arange(len(n_words_list))  # the x locations for the groups
    width = 0.35  # the width of the bars

    accuracies = [results[(n_grams[0], similarity_metrics[0], cl)][0] for cl in n_words_list]
    f1_scores = [results[(n_grams[0], similarity_metrics[0], cl)][1] for cl in n_words_list]

    rects1 = ax.bar(ind - width/2, accuracies, width, label='Accuracy')
    rects2 = ax.bar(ind + width/2, f1_scores, width, label='F1 Score')

    ax.set_xlabel('Concatenated Sentences')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by number of concatenated sentences')
    ax.set_xticks(ind)
    ax.set_xticklabels([str(cl) for cl in n_words_list])
    ax.legend()
    plt.show()

    def autolabel(rects):
            """Attach a text label above each bar displaying its height"""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()

    # Print summary of results
    print("\nSummary of Results:")
    for (n, metric, n_words), (accuracy, f1) in results.items():
        print(f"N-gram: {n}, Metric: {metric}, #n words: {n_words} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")


def log_results(n_gram, metric, n_words, accuracy, f1):
    log_message = f"N-gram: {n_gram}, Metric: {metric}, #n words: {n_words}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}"
    print(log_message)  # Print to console
    # Save to a log file
    with open('evaluation_log.txt', 'a') as log_file:
        log_file.write(log_message + "\n")

        
        
def create_fixed_word_sentences(sentences, n_words):
    """Create sentences with exactly 'n_words' words from the given sentences."""
    fixed_sentences = []
    current_words = []
    
    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)
        
        while len(current_words) >= n_words:
            fixed_sentences.append(' '.join(current_words[:n_words]))
            current_words = current_words[n_words:]
    
    return fixed_sentences