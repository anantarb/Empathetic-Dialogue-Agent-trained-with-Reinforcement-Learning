"""
This file implements different metric calculation functions for a given conversation.
"""
# Note: some imports are defined inside the methods of the metrics (for cases where only some metrics are computed)
import os
import string
from pathlib import Path
import torch
import re
import numpy as np
from nltk.corpus import stopwords
import sys
sys.path.append("..")


stopwords = stopwords.words('english')
question_words = {'who', 'what', 'why', 'where', 'how', 'when'}
_ = [stopwords.remove(q) for q in question_words]
punct = list(string.punctuation)
contractions = ["'s", "'d", "'ld", "n't", "'re", "'ll", "'ve"]
filters = set(stopwords + contractions + punct)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / np.sqrt((np.sum(a * a, axis=1) * np.sum(b * b, axis=1)))

def removeContradictions(wordlist):
    p = re.compile('|'.join(map(re.escape, contractions)))
    ret = [p.sub('', s) for s in wordlist]
    return set(ret)

def question(conversation):
    """Counts whether each utterance in the given conversation contains a question (yes: 100, no: 0)"""
    num_turns = len(conversation)

    if any(question_word in conversation[-1] for question_word in question_words) or '?' in conversation[-1]:
        return 1
    return 0

def questionSampVsGold(gold_utterance, sample_utterance):
    """Counts whether each utterance in the given conversation contains a question (yes: 100, no: 0)"""
    gold_utterance_question = False
    sample_utterance_question = False
    if any(question_word in gold_utterance for question_word in question_words) and '?' in gold_utterance:
        gold_utterance_question = True
    if any(question_word in sample_utterance for question_word in question_words) and '?' in sample_utterance:
        sample_utterance_question = True
    if sample_utterance_question == gold_utterance_question:
        return 1
    return 0


def conversation_repetition(conversation):
    """Counts the number of distinct words in the current utterance that were also in any of the previous utterances"""
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    # filter stopwords, contractions and punctuation
    filtered = [set(utterance).difference(filters)
                for utterance in conversation]

    for i in range(1, num_turns):
        current = removeContradictions(filtered[i])
        prev = removeContradictions(set.union(*filtered[:i]))
        repeats = current.intersection(prev)
        num_repeats_in_utterances[i] = len(repeats)

    return num_repeats_in_utterances[-1]


# called 'reward_conversation_repetition' (for bot utterances) in original paper
def self_repetition(conversation):
    """
    Counts the number of distinct words in the current utterance that were also in any of the previous utterances of
    the current speaker (assuming two-speaker multi-turn dialog)
    """
    #print('self_repetition')
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    # filter stopwords, contractions and punctuation
    filtered = [set(utterance).difference(filters)
                for utterance in conversation]

    # first and second utterance can't repeat any word of previous utterances of the current speaker
    num_repeats_in_utterances[0] = 0
    num_repeats_in_utterances[1] = 0
    for i in range(2, num_turns):
        current = removeContradictions(filtered[i])  # current utterance
        # all utterances of the current speaker so far
        prev = removeContradictions(set.union(*filtered[:i][i % 2::i]))
        repeats = current.intersection(prev)
        num_repeats_in_utterances[i] = len(repeats)

    return num_repeats_in_utterances[-1]


# called 'word_similarity' in original paper
def utterance_repetition(conversation):
    """Counts the number of distinct words in the current utterance that were also in the previous utterance"""
    num_turns = len(conversation)
    num_repeats_in_utterances = np.zeros(num_turns)

    filtered = [set(utterance).difference(filters) for utterance in
                conversation]  # filter stopwords, contractions and punctuation

    # first utterance can't repeat any word of previous utterance
    num_repeats_in_utterances[0] = 0
    for i in range(1, num_turns):
        current = removeContradictions(filtered[i])
        prev = removeContradictions(filtered[i-1])
        repeats = current.intersection(prev)
        num_repeats_in_utterances[i] = len(repeats)

    return num_repeats_in_utterances[-1]


def word_repetition(conversation):  # called 'utterance_repetition' in original paper
    """Counts the number of words that occur multiple times within the same utterance (duplicates) """

    filtered = [token for token in conversation[-1] if token not in filters]
    # filter stopwords, contractions and punctuation

    # (difference is positive if a word occurs multiple times)
    repeats = len(filtered) - len(set(filtered))
    num_repeats_in_utterances = repeats

    return num_repeats_in_utterances


def utterance_length(conversation):
    """Counts the length of each utterance."""
    filtered = [token for token in conversation[-1] if token not in punct]
    # filter punctuation

    return len(filtered)


def empathy(conversation):
    """
    Computes the levels of the three empathy mechanisms (emotional reactions, interpretations, explorations) of
    each utterance with respect to its previous utterance.
    """
    # Init empathy_classifier just once
    if 'empathy_classifier' not in globals():
        # load the empathy classifier
        print('\nLoading empathy_classifier')

        # workaround: import EmpathyClassifier here
        from empathy_mental_health.api.empathy import EmpathyClassifier
        with torch.no_grad():
            global empathy_classifier
            empathy_classifier = EmpathyClassifier(run_on_cpu=True)

    utterances = np.array([' '.join(tokens) for tokens in conversation[-2:]])

    utterance_pairs = np.vstack(
        (np.roll(utterances, shift=1, axis=0), utterances)).T
    # for the first utterance the empathy with the previous utterance cannot be computed
    # --> delete the first pair and add zero empathy level for it later (for all three empathy communication mechanisms)
    utterance_pairs = np.delete(utterance_pairs, 0, 0)

    #print(utterance_pairs)

    # Run empathy_classifier: get empathy levels for each utterance pair for each mechanism
    empathy_levels = empathy_classifier.compute_empathy_levels(
        utterance_pairs)  # shape: (utterance_pair, mechanism)

    # set empathy levels for the first utterance to zero (for all three empathy communication mechanisms)
    empathy_levels = np.vstack((np.zeros(3).astype(int), empathy_levels))

    #list = empathy_levels[1]
    #dict = {"emotional_reaction_level":list[0],"interpretation_level":list[1],"exploration_level":list[2]}
    return empathy_levels[1]  # shape: (utterance, empathy levels)





