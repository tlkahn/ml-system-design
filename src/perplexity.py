# [[file:../README.org::*Footnotes][Footnotes:1]]
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import word_tokenize, sent_tokenize
from math import exp

# Sample text
text = "This is a sample text. We use it to demonstrate perplexity calculation."

# Tokenize and preprocess
sentences = sent_tokenize(text)
tokenized_text = [word_tokenize(sent) for sent in sentences]
n = 3  # Trigram model
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

# Train language model
model = MLE(n)
model.fit(train_data, padded_sents)

# Test sentence
test_sentence = "This is a test sentence."
tokenized_test_sentence = word_tokenize(test_sentence)

# Calculate perplexity
test_ngrams = list(nltk.ngrams(tokenized_test_sentence, n))
log_prob = 0

for ngram in test_ngrams:
    if model.score(ngram[:-1], ngram[-1]) > 0:
        log_prob += -1 * model.logscore(ngram[-1], ngram[:-1])

perplexity = exp(log_prob / len(test_ngrams))
print(f"Perplexity: {perplexity}")
# Footnotes:1 ends here
