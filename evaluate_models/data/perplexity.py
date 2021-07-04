import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import csv
from nltk.lm import Vocabulary


n = 2
def prepareTestData():  
    train_sentences = getTestSentences()
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                    for sent in train_sentences]

    train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    # unigram
    #    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    words = [word for sent in tokenized_text for word in sent]
    words.extend(["<s>", "</s>"])
    padded_vocab = Vocabulary(words)
    model = MLE(n)
    model.fit(train_data, padded_vocab)
    return model

def getTestSentences():
    train_sentences = []
    with open('datasets/test.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_sentences.append(row['utterance'])
    return train_sentences

def getAveragePerplexity(model, test_sentences):
    n = 1
    #test_sentences = ['an apple', 'an ant']
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                    for sent in test_sentences]

    #test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]

    #for test in test_data:
    #    print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    #test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    values = []
    for i, test in enumerate(test_data):
        perplexity = model.perplexity(test)
        #print("PP({0}):{1}".format(test_sentences[i], perplexity))
        if perplexity != float('+inf'):
            values.append(perplexity)
        else:
            values.append(1000000)
    #print(values)
    avg = sum(values)/len(values)
    return avg