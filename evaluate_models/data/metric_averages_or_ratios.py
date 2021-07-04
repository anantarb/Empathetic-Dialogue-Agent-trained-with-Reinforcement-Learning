
import multiprocessing
from datetime import datetime
import time
import numpy as np
from joblib import Parallel, delayed
from . import metrics

num_cores = multiprocessing.cpu_count()

def getQuestionVsGoldAvg(parsed_conversations_gold,parsed_conversations_samples):
    questionComp = list()
    for i in range(len(parsed_conversations_gold)):
        questComp = metrics.questionSampVsGold(parsed_conversations_gold[i][-1],parsed_conversations_samples[i][-1])
        questionComp.append(questComp)
    avg = sum(questionComp) / len(questionComp)
    return avg

def getSampleQuestionOfAllSamplesRatio(parsed_conversations_samples):
    numberOfSampleQuestions = 0
    for i in range(len(parsed_conversations_samples)):
        isSampleQuestion = metrics.question(parsed_conversations_samples[i][-1])
        if (isSampleQuestion):
            numberOfSampleQuestions += 1
    ratio = numberOfSampleQuestions / len(parsed_conversations_samples)
    return ratio

def getGoldQuestionVsSampleRatio(parsed_conversations_gold,parsed_conversations_samples):
    numberOfGoldQuestions = 0
    numberOfSampleQuestionsIfGoldQuestion = 0
    for i in range(len(parsed_conversations_gold)):
        isSampleQuestion = metrics.question(parsed_conversations_samples[i][-1])
        isGoldQuestion = metrics.question(parsed_conversations_gold[i][-1])
        if (isGoldQuestion):
            numberOfGoldQuestions += 1
            if (isSampleQuestion):
                numberOfSampleQuestionsIfGoldQuestion += 1
    ratio = numberOfSampleQuestionsIfGoldQuestion / numberOfGoldQuestions
    return ratio

def getNoGoldQuestionVsSampleRatio(parsed_conversations_gold,parsed_conversations_samples):
    numberOfNoGoldQuestions = 0
    numberOfSampleQuestionsIfGoldQuestion = 0
    for i in range(len(parsed_conversations_gold)):
        isSampleQuestion = metrics.question(parsed_conversations_samples[i][-1])
        isGoldNoQuestion = (metrics.question(parsed_conversations_gold[i][-1]) == 0)
        if (isGoldNoQuestion):
            numberOfNoGoldQuestions += 1
            if (isSampleQuestion):
                numberOfSampleQuestionsIfGoldQuestion += 1
    ratio = numberOfSampleQuestionsIfGoldQuestion / numberOfNoGoldQuestions
    return ratio

def getMetrics(conversation, metric_names):
    conversation_metrics = []

    for metric in metric_names:
        metric_calculator = getattr(metrics, metric)  # get metric calculation function
        cur_metric_for_conv = metric_calculator(conversation)  # apply metric calcuation function
        if metric == 'empathy':
            conversation_metrics.append(cur_metric_for_conv[0])  # emotional_reaction_level in the conversation
            conversation_metrics.append(cur_metric_for_conv[1])  # interpretation_level in the conversation
            conversation_metrics.append(cur_metric_for_conv[2])  # exploration_level in the conversation
        else:
            conversation_metrics.append(cur_metric_for_conv)
    return conversation_metrics


def getMetricDict(parsed_conversations, metric_names, content_metric_names_separated):
    metric_list = []

    t = time.process_time()
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    for i in range(len(parsed_conversations)):
        metric_list.append(getMetrics(parsed_conversations[i], metric_names))
        #if (i%500)==0:
        #    print("Calculated metric for %i dialogs" % i)

    elapsed_time = time.process_time() - t
    print(elapsed_time)

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
 
    all_metrics = np.array(metric_list)
    
    mean_metric = all_metrics.mean(axis=0)
    results = dict(zip(content_metric_names_separated, mean_metric))
    return results