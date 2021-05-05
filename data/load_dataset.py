
import numpy as np


def sentence_to_tensor(context, response, encoder, maxlen):
    """
    Parameters
    ----------
    context:
        context in the input 
    response:
        response to the context
    encoder:
        GPT-2 encodings
    maxlen: 
        Maximum length of the input
    """
    
    context_tokens = encoder.encode(context)
    start_of_conv = encoder.encode(" <SOC> ")
    end_of_text = encoder.encode(" <EOT> ")
    response_tokens = encoder.encode(response)
    input_tokens_len = len(context_tokens) + len(response_tokens) + 10
    padding = []
    if input_tokens_len >= maxlen:
        if len(context_tokens) > input_tokens_len - maxlen:
            # cut context from beginning if length of context + response is too long
            # and len of context is long enough to cut
            context_tokens = context_tokens[input_tokens_len - maxlen:]
        else:
            # cut response from end if length of context + response is too long
            # and len of response is long enough to cut
            if maxlen-len(context_tokens)-10 < 0:
                return None
            response_tokens = response_tokens[:maxlen-len(context_tokens)-10]
    else:
        remaining_length = maxlen - input_tokens_len
        padding = encoder.encode("#") * remaining_length
    
    tokens = context_tokens + start_of_conv + response_tokens + end_of_text + padding
    
    return tokens
    



# function to pre-process training data
def process_training_data(source_path, encoder, maxlen=100, history_len=1, reactonly=False):
    """
    Parameters
    ----------
    source_path:
        path where the csv file is located 
    encoder: 
        GPT-2 encodings
    maxlen: 
        Maximum length of the input
    history_len: 
        how many previous dialogues do we want
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
                tokens = sentence_to_tensor(prev_str, sent, encoder, maxlen)
                if tokens:
                    data.append(np.stack(tokens))
                
        else:
            history = []
    
    return data




