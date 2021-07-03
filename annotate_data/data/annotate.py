import json
import random
import time
import os
import sys
import getopt

from itertools import combinations
from pathlib import Path
from typing import Text

from models import encodings


class bcolors:
    TEXT = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class config:
    size_of_each_dialog_line = 50  # specify printing format
    space_between_dialogs = 10  # specify printing format


def get_formated_annotator_decision(annotator_decision):
    # take random sample if both are equally good (3)
    if annotator_decision == '3':
        return (random.randint(0, 1) and '0' or '1')
    # take 0 if first (left sample) is better, take 1 if second (right) is better
    if annotator_decision == '1':
        return '0'
    if annotator_decision == '2':
        return '1'
    return annotator_decision


def format_context(query):
    ret_str = []
    list = query.split('<SOC>')
    for i in range(len(list)):
        col_line = f"{bcolors.TEXT}" + \
            ((i % 2 == 0) and 'A' or 'B') + ': ' + list[i] + f"{bcolors.ENDC}"
        ret_str.append(col_line)
    return "\n".join(ret_str)


def make_lines_of_same_size(text, size):
    text_with_lines_of_same_size = ''
    for line in text.splitlines():
        if len(line) > size:  # line is longer than specified size --> split line in multiple lines of equal length
            for newline_startid in range(0, len(line), size):
                if newline_startid == 0:  # add first part of line
                    text_with_lines_of_same_size += (
                        line[newline_startid:newline_startid + size] + '\n')
                else:
                    # 1. first line contains speaker id '[1]: ' or '[2]: ' --> add whitespace to other lines '     '
                    # 2. new line might start with whitespace --> remove left whitespace of newline using lstrip()
                    # 3. new line might be smaller than specified size --> fill new line with whitespace using ljust()
                    text_with_lines_of_same_size += ((line[newline_startid:newline_startid + size].lstrip())
                                                     .ljust(size)
                                                     + '\n')
        else:  # line might be smaller than specified size --> fill end of line with whitespace using ljust()
            text_with_lines_of_same_size += (line.ljust(size) + '\n')
    return text_with_lines_of_same_size


def show_instructions():
    print("\nWelcome to the dialog annotation tool!\n"
          "During the next few minutes you will be presented with "
          "pairs of responses to the same dialog history, side-by-side, for which you should "
          "decide which response is better.\n"
          "You will have the following options:\n"
          f"Type {bcolors.OKGREEN}1{bcolors.ENDC} for left,\n"
          f"     {bcolors.OKGREEN}2{bcolors.ENDC} for right,\n"
          f"     {bcolors.OKGREEN}3{bcolors.ENDC} if they are equally good,\n"
          f"     {bcolors.OKGREEN}4{bcolors.ENDC} if they are uncomparable, or\n"
          f"     {bcolors.OKGREEN}5{bcolors.ENDC} if left is better but you want to only use part of the response\n"
          f"     {bcolors.OKGREEN}6{bcolors.ENDC} if right is better but you want to only use part of the response\n"
          f"     {bcolors.FAIL}exit{bcolors.ENDC} to stop annotation.\nHit enter to save your decision.\n"
          f"\nThe annoation criteria are:\n"
          f"1. Empathy/Sympathy: Which response shows understanding of the feelings of the person talking about their experience?\n"
          f"2. Relevance: Which response seems appropriate to the conversation? Was it on-topic?\n"
          f"3. Fluency: Which response seems more fluent and accurate?\n"
          f"\nThe next dialog and response pair will automatically be shown to you after you hit enter.")


def show_dialog_intro(annotator, i, len, query):
    col_query = format_context(query)
    state_str = ("Annotator {annotator}: [Sample {current} of {all}]").format(
        annotator=annotator, current=str(i+1), all=str(len))
    print(f"{bcolors.TEXT}%s{bcolors.ENDC}" % state_str)
    print((col_query.replace('<SOC>', '\n'))+'\n'+'-' *
          (2*config.size_of_each_dialog_line+config.space_between_dialogs))
    print('\nWhich next turn is better?\n' + '-' *
          (2*config.size_of_each_dialog_line+config.space_between_dialogs))


def print_dialogs_side_by_side(a, b, size=50, space=4):
    separator = int(space/2)*' ' + '|' + int(space/2)*' '
    print('Dialog 1'.ljust(size) + separator + 'Dialog 2')

    # make sure all lines in a and b have the same length
    a = make_lines_of_same_size(text=a, size=size)
    b = make_lines_of_same_size(text=b, size=size)

    # combine each line of left and right dialog using the separator and print the resulting line
    lines_a = a.splitlines()
    lines_b = b.splitlines()
    appending_line = ' '*size

    # fill up the shorter sample
    if (len(b.splitlines()) > len(a.splitlines())):
        diff = len(b.splitlines()) - len(a.splitlines())
        x = a.splitlines()
        lines_a = x + [appending_line] * diff
        lines_b = b.splitlines()
    if (len(a.splitlines()) > len(b.splitlines())):
        diff = len(a.splitlines()) - len(b.splitlines())
        x = b.splitlines()
        lines_b = x + [appending_line] * diff
        lines_a = a.splitlines()

    lines = [lines_a, lines_b]

    for l in zip(*lines):
        l = map(lambda line_ele: f"{bcolors.TEXT}" +
                line_ele+f"{bcolors.ENDC}", l)
        print(*l, sep=separator)


def encode_and_pad(response):
    checkpoint_path = 'gpt-2/models/124M'
    enc = encodings.Encoding("main", n_vocab=50257, eot_token=50256, base_path=checkpoint_path).get_encoder()
    encoded = enc.encode(response)
    if not (encoded[len(encoded)-1] == int(198)):
        encoded = encoded + [int(198)]
    pad_number = 100 - len(encoded)
    if pad_number > 0:
        encoded = encoded + [50259] * pad_number
    if pad_number < 0:
        encoded = encoded[0:100]
    
    assert len(encoded) == 100
    return encoded


def annotate(inputfile):
    input_path = inputfile
    decoded_input_path = 'datasets/distributed_data/decoded_distributed_data.json'
    with open(input_path) as json_file:
        data = json.load(json_file)
    with open(decoded_input_path) as json_file:
        data_decoded = json.load(json_file)
    names = [*data_decoded]
    name = input("What's your name?\n")
    if name not in names:
        print("Sorry! Your name is not in the annotators list. The annotators are: ", names)
        print("Please try again or tell project owners to put your name in the annotators list.")
        return
    
    if not os.path.isdir('./datasets/annotated'):
        os.mkdir('./datasets/annotated')
    
    output_path = 'datasets/annotated/' + name + ".json"
    decoded_output_path = 'datasets/annotated/' + 'decoded_' + name + ".json"

    if os.path.exists(output_path):
        with open(output_path) as json_file:
            output_data = json.load(json_file)
        with open(decoded_output_path) as json_file:
            output_data_decoded = json.load(json_file)
        counter = output_data_decoded['counter']
        print("Resuming Session")
    else:
        counter = 0
        output_data = {}
        output_data[name] = []
        output_data_decoded = {}
        output_data_decoded[name] = []

    starting_sample = counter
    parsed_comparisons = []

    # read in dialog
    for sample in data_decoded[name]:
        query = sample['query']
        # choose random order of the two comparisons
        sample0 = sample['sample0']
        sample1 = sample['sample1']
        parsed_comparisons.append(({"Q": query, "S0": sample0, "S1": sample1}))

    show_instructions()

    while True:
        print(f'\nIf you are ready, type {bcolors.OKGREEN}start{bcolors.ENDC}.\n'
              f'If you want to quit, type {bcolors.FAIL}exit{bcolors.ENDC}.\n'
              f"Output in file %s.\n"
              % decoded_output_path)
        start_or_exit = input("Your input: ")
        if start_or_exit == 'exit':
            return
        if start_or_exit == 'start':
            print(f'{bcolors.OKGREEN}-{bcolors.ENDC}' *
                  (2*config.size_of_each_dialog_line+config.space_between_dialogs))
            break

    exit_typed = False
    for i in range(starting_sample, len(parsed_comparisons)):
        print("This is the dialog history:\n"+'-' *
              (2*config.size_of_each_dialog_line+config.space_between_dialogs))

        query = parsed_comparisons[i]["Q"]
        left_response = parsed_comparisons[i]["S0"]
        right_response = parsed_comparisons[i]["S1"]
        show_dialog_intro(name, i, len(parsed_comparisons), query)
        print_dialogs_side_by_side(left_response, right_response,
                                   size=config.size_of_each_dialog_line,
                                   space=config.space_between_dialogs)
        print('-'*(2*config.size_of_each_dialog_line+config.space_between_dialogs))
        while True:
            print("\nWhich dialog is better? "
                  f"Type {bcolors.OKGREEN}1{bcolors.ENDC} for left, "
                  f"{bcolors.OKGREEN}2{bcolors.ENDC} for right, "
                  f"{bcolors.OKGREEN}3{bcolors.ENDC} if they are equally good, "
                  f"{bcolors.OKGREEN}4{bcolors.ENDC} if they are uncomparable "
                  f"{bcolors.OKGREEN}5{bcolors.ENDC} if left is better but you want to only use part of the response "
                  f"{bcolors.OKGREEN}6{bcolors.ENDC} if right is better but you want to only use part of the response "
                  f"(or {bcolors.FAIL}exit{bcolors.ENDC} to stop annotation) and hit enter.\n")
            annotator_decision = input("Your input: ")
            if annotator_decision == 'exit':
                exit_typed = True
                break

            if annotator_decision in ['1', '2', '3', '4', '5', '6']:
                best = get_formated_annotator_decision(annotator_decision)
                if best != '4':
                    temp = {}
                    temp['query'] = data_decoded[name][i]['query']
                    # copy only good part of response
                    if annotator_decision == '5':
                        new_left = input("Copy good part from left sample: ")
                        temp['sample0'] = new_left
                        temp['sample1'] = data_decoded[name][i]['sample1']
                        best = '0'
                    elif annotator_decision == '6':
                        new_right = input("Copy good part from right sample: ")
                        temp['sample0'] = data_decoded[name][i]['sample0']
                        temp['sample1'] = new_right
                        best = '1'
                    else:
                        temp['sample0'] = data_decoded[name][i]['sample0']
                        temp['sample1'] = data_decoded[name][i]['sample1']
                    temp['best'] = int(best)
                    output_data_decoded[name].append(temp)
                    output_data_decoded['counter'] = i + 1

                    temp1 = {}
                    temp1['query'] = data[name][i]['query']
                    if annotator_decision == '5':
                        temp1['sample0'] = encode_and_pad(new_left)
                        temp1['sample1'] = data[name][i]['sample1']
                        best = '0' 
                    elif annotator_decision == '6':
                        temp1['sample0'] = data[name][i]['sample0']
                        temp1['sample1'] = encode_and_pad(new_right)
                        best = '1'
                    else:
                        temp1['sample0'] = data[name][i]['sample0']
                        temp1['sample1'] = data[name][i]['sample1']
                    temp1['best'] = int(best)

                    output_data[name].append(temp1)
                    with open(output_path, 'w') as outfile:
                        json.dump(output_data, outfile)

                    with open(decoded_output_path, 'w') as outfile:
                        json.dump(output_data_decoded, outfile)

                else:
                    temp = {}
                    temp['query'] = data_decoded[name][i]['query']
                    temp['sample0'] = data_decoded[name][i]['sample0']
                    temp['sample1'] = data_decoded[name][i]['sample1']
                    temp['best'] = int(best)
                    output_data_decoded[name].append(temp)
                    output_data_decoded['counter'] = i + 1
                    with open(decoded_output_path, 'w') as outfile:
                        json.dump(output_data_decoded, outfile)

                break

            print("Sorry, your input did not match any of "
                  f"{bcolors.OKGREEN}1{bcolors.ENDC}, "
                  f"{bcolors.OKGREEN}2{bcolors.ENDC}, "
                  f"{bcolors.OKGREEN}3{bcolors.ENDC}, "
                  f"{bcolors.OKGREEN}4{bcolors.ENDC} or "
                  f"{bcolors.FAIL}exit{bcolors.ENDC}.")
        if exit_typed:
            break
        print(f'\nAnnotation finished successfully. \nYou can find your annotation results under: %s'
              % str(decoded_output_path))
    return