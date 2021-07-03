import numpy as np
import random
import json

# distribute data returns data which is not decoded by byte pair encodings
# the data will be in the form of numbers
def distribute_data(data_path, annotators, same_data, total_for_each, seed=245):
    
    # same_data represents how many same data is on each annotators quotas (for comparision purpose)
    # total_for_each represents how many data shound be on each annotators quotas excluding same_data


    with open(data_path) as json_file:
        data = json.load(json_file)

    random.seed(seed)
    random.shuffle(data)

    distributed_data = {}
    track_index = 0
    for name in annotators:
        first_data = data[:same_data]
        next_data = first_data + data[track_index + same_data: track_index + same_data + total_for_each]
        distributed_data[name] = next_data
        track_index = track_index + (len(next_data) - same_data)
    distributed_data['counter'] = track_index + same_data

    return distributed_data
    
# decode distribute data returns data which is decoded by byte pair encodings
# the data will be in the form of strings
def decode_distributed_data(distributed_data, encoder):
    decoded_data = {}
    for annotators in distributed_data:
        if annotators == "counter":
            continue
        for i in range(len(distributed_data[annotators])):
            if annotators not in decoded_data:
                decoded_data[annotators] = []
            temp = {}
            temp["query"] = encoder.decode(distributed_data[annotators][i]["query"])
            temp["sample0"] = encoder.decode(distributed_data[annotators][i]["sample0"])
            temp["sample1"] = encoder.decode(distributed_data[annotators][i]["sample1"])
            decoded_data[annotators].append(temp)

    return decoded_data
