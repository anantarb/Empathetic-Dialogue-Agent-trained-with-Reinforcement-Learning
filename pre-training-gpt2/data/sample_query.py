import numpy as np
import random
import json

# see process_data.py for more info
# Here, we do exact same thing as in process_data.py
# except, we just sample context but not response

def context_to_tensor(context, encoder, context_maxlen):
    
    #with open('datasets/context_prefix.json') as json_file:
    #    context_prefix = json.load(json_file)
    
    #context_tokens = encoder.encode(context_prefix['prefix']) + encoder.encode(context)
    context_tokens = encoder.encode(context)
    padding_token = encoder.padding_token
    
    
    if len(context_tokens) < context_maxlen:
        context_tokens =  context_tokens + [padding_token] * (context_maxlen - len(context_tokens))
    else:
        context_tokens =  context_tokens[len(context_tokens)-context_maxlen:]
    
    tokens = context_tokens
    
    assert len(tokens) == context_maxlen
    
    return tokens
    
def sample_query(source_path, encoder, context_maxlen=100, history_len=1):
    
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
            context = " <SOC> ".join(history[-max_hist_len :])
            tokens = context_to_tensor(context, encoder, context_maxlen)
            if tokens:
                data.append(np.stack(tokens))
                
        else:
            history = []
    
    return data
