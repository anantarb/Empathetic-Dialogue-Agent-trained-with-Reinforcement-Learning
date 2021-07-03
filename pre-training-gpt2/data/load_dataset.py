import random
import numpy as np


def sentence_to_tensor(context, response, encoder, context_maxlen, response_maxlen):
    """
    Parameters
    ----------
    context:
        context in the input 
    response:
        response to the context
    encoder:
        GPT-2 encodings
    context_maxlen: 
        Maximum length of the context
    response_maxlen:
        Maximum length of the response
        
    Returns:
        Byte pair encoding of the input (context + response)    
    """
    
    context_tokens = encoder.encode(context)
    start_of_conv = encoder.encode(" </s> ")
    end_of_reponse = encoder.encode("\n")
    response_tokens = encoder.encode(response) 
    
    padding_token = encoder.padding_token
    
    context_tokens = context_tokens[:context_maxlen]
    response_tokens = response_tokens[:response_maxlen] 
    
    if len(context_tokens) < context_maxlen:
        context_tokens =  context_tokens + [padding_token] * (context_maxlen - len(context_tokens))
        
    if len(response_tokens) < response_maxlen:
        response_tokens = response_tokens + end_of_reponse + [padding_token] * (response_maxlen - 1 - len(response_tokens))
    
    tokens = context_tokens + start_of_conv + response_tokens
    
    assert len(tokens) == context_maxlen + response_maxlen + len(start_of_conv)
    
    return tokens
    



# function to pre-process training data
def process_training_data(source_path, encoder, context_maxlen=100, response_maxlen=100, history_len=1, reactonly=False):
    """
    Parameters
    ----------
    source_path:
        path where the csv file is located 
    encoder: 
        GPT-2 encodings
    context_maxlen: 
        Maximum length of the context
    response_maxlen:
        Maximum length of the response
    history_len: 
        how many previous dialogues do we want in context
    reactonly: 
        If set true, only parses one dialogue; useful for testing
    
    Returns
    -------
        a list with input data i.e tokens of the input needed to train the model

    """

    # open the file
    df = open(source_path).readlines()
    max_hist_len = history_len
    data = []
    history = []
    for i in range(1, len(df)):
        cparts = df[i - 1].strip().split(",")
        sparts = df[i].strip().split(",")
        if cparts[0] == sparts[0]:
            prevsent = cparts[5].replace("_comma_", ",").strip()
            history.append(prevsent)
            idx = int(sparts[1])
            if not reactonly or ((idx % 2) == 0):
                prev_str = " <SOC> ".join(history[-max_hist_len :])
                sent = sparts[5].replace("_comma_", ",").strip()
                tokens = sentence_to_tensor(prev_str, sent, encoder, context_maxlen, response_maxlen)
                if tokens:
                    data.append(np.stack(tokens))
                
        else:
            history = []
    
    return data

# Batch sampler class that samples batch of data from the list given by process_training_data function
class Batch_Sampler(object):

    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.total_size = len(data)
        self.batch_size = batch_size
        self.current_index = 0
        if shuffle:
            random.shuffle(data)

    def sample(self):
        if self.current_index + self.batch_size > self.total_size:
            self.current_index = 0
        prev_index = self.current_index
        self.current_index = self.current_index + self.batch_size
        return self.data[prev_index:self.current_index]


