
import json
import pprint
import re
import numpy as np

def splitWords(sentence):
    s = sentence.replace('.','').strip()
    sent_list = re.split(' |,|;',s)
    return sent_list

def getParsedConversations(file_path,response_type):
    # read dialog data
    with open(file_path) as f:
        file = json.load(f)

    parsed_conversations = list()

    # read in dialog
    for sample in file:
        query = sample['query'].lower()
        sample = sample[response_type].lower()
        if (sample.strip() == ''):
            #print("empty")
            sample = 'none'
        conversation = query.split('<soc>')
        conversation = list(map(splitWords,conversation))
        conversation.append(splitWords(sample))
        parsed_conversations.append(conversation)
    print('Done! Parsed %i dialogs' % (len(parsed_conversations)))
    return parsed_conversations

def getGeneratedUtterances(file_path, response_type):
    # read dialog data
    with open(file_path) as f:
        file = json.load(f)

    parsed_conversations = list()

    # read in dialog
    for sample in file:
        sample = sample[response_type].lower()
        #print("The query: "+query)
        #print("The sample: >"+sample+"<")
        #print(sample=='')
        if (sample.strip() == ''):
            #print("empty")
            sample = 'none'
        parsed_conversations.append(sample)
    print('Done! Parsed %i dialogs' % (len(parsed_conversations)))
    return parsed_conversations


def removeAllSigns(token):
    token1 = token.replace(".","")
    token2 = token1.replace(",","")
    token3 = token2.replace("?","")
    token4 = token3.replace("!","")
    return token4

def getParsedConversationsGoldVsSample(file_path, file_path_gold):
    # read dialog data
    with open(file_path) as f:
        file = json.load(f)
    with open(file_path_gold) as f:
        gold_file = json.load(f)

    parsed_comparisons = list()

    # read in dialog
    for i in range(len(file)):
        gold = removeAllSigns(gold_file[i]['gold_response'].lower())
        try:
            sample = removeAllSigns(file[i]['sample0'].lower())
        except KeyError:
            sample = removeAllSigns(file[i]['sample'].lower())
        parsed_comparisons.append({'gold':gold,'sample':sample})
    print('Done! Parsed %i dialogs' % (len(parsed_comparisons)))
    return parsed_comparisons