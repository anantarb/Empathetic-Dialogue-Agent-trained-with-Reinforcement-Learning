import json
import random
import time
import os
import sys, getopt

from itertools import combinations
from pathlib import Path
from typing import Text


class bcolors:
    TEXT = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'




def make_lines_of_same_size(text, size):
    text_with_lines_of_same_size = ''
    for line in text.splitlines():
        if len(line) > size:  # line is longer than specified size --> split line in multiple lines of equal length
            for newline_startid in range(0, len(line), size):
                if newline_startid == 0:  # add first part of line
                    text_with_lines_of_same_size += (line[newline_startid:newline_startid + size] + '\n')
                else:
                    # 1. first line contains speaker id '[1]: ' or '[2]: ' --> add whitespace to other lines '     '
                    # 2. new line might start with whitespace --> remove left whitespace of newline using lstrip()
                    # 3. new line might be smaller than specified size --> fill new line with whitespace using ljust()
                    text_with_lines_of_same_size += (('     ' + line[newline_startid:newline_startid + size].lstrip())
                                                     .ljust(size)
                                                     + '\n')
        else:  # line might be smaller than specified size --> fill end of line with whitespace using ljust()
            text_with_lines_of_same_size += (line.ljust(size) + '\n')
    return text_with_lines_of_same_size

def format_context(query):
    ret_str = []
    list = query.split('<SOC>')
    for i in range(len(list)):
        col_line = f"{bcolors.TEXT}" + ((i%2==0) and 'A' or 'B' ) + ': ' + list[i] + f"{bcolors.ENDC}"
        ret_str.append(col_line)
    return "\n".join(ret_str)



def print_dialogs_side_by_side(a, b, size=50, space=4):
    separator = int(space/2)*' ' + '|' + int(space/2)*' '
    print('Dialog 1'.ljust(size) + separator + 'Dialog 2')

    # make sure all lines in a and b have the same length
    a = make_lines_of_same_size(text=a, size=size)
    b = make_lines_of_same_size(text=b, size=size)

    # combine each line of left and right dialog using the separator and print the resulting line
    lines = [a.splitlines(), b.splitlines()]
    for l in zip(*lines):
        l = map(lambda line_ele: f"{bcolors.TEXT}"+line_ele+f"{bcolors.ENDC}", l)
        print(*l, sep=separator)


def create_missing_directories(results_path):
    if not os.path.exists(results_path):
        print('Top level results directory missing. Creating the missing directory ...')
        os.makedirs(results_path)
        print('Results directory successfully created.')


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
          f"     {bcolors.FAIL}exit{bcolors.ENDC} to stop annotation.\nHit enter to save your decision."
          f"\nThe next dialog and response pair will automatically be shown to you after you hit enter.")


def annotate(inputfile='', outputfile=''):
    continue_old_file = True
    if outputfile == '':
        continue_old_file = False
    print ('Input file is "', inputfile)
    print ('Output file is "', outputfile)

    # CONFIGURATION
    project_dir = Path(__file__).resolve().parent
    if inputfile == "":
        data_filepath = project_dir.joinpath('../data/test_comparisons_output.json')
    else:
        data_filepath = project_dir.joinpath(inputfile)

    num_comparisons_per_dialog_length = 10  # create n comparisons per dialog length
    size_of_each_dialog_line = 50  # specify printing format
    space_between_dialogs = 10  # specify printing format
    random.seed(7)
    # CONFIGURATION END

    results_dir = project_dir.joinpath('results')

    # setup
    # file management
    if not continue_old_file:
        create_missing_directories(results_dir)
        current_timestamp = time.strftime("%d-%m-%Y-%H-%M", time.gmtime())
        outputfile = '%s_%s.json' % ('annotated_samples', current_timestamp)

    results_filepath = results_dir.joinpath(outputfile)

    starting_sample = 0

    if continue_old_file:
        # load data from json file
        with open(results_filepath) as f:
            annotated_data = json.load(f)
            starting_sample = annotated_data['counter']


    # setup end

    # read dialog data
    with open(data_filepath) as f:
        data = json.load(f)

    parsed_comparisons = []

    # create a mapping from dialog indices to lengths and scores (ids_per_length)
    # for selecting desired dialog pairs later
    for sample in data:
        query = sample['query']
        # choose random order of the two comparisons
        bool = random.randint(0,1)
        sample0 = ( bool and sample['sample0'] or sample['sample1'] )
        sample1 = ( bool and sample['sample1'] or sample['sample0'] )
        parsed_comparisons.append(({"Q":query,"S0":sample0,"S1":sample1}))
        print('Done! Parsed %i dialogs' % (len(parsed_comparisons)))

    # START INTERACTION
    # show welcome instructions
    show_instructions()

    while True:
        start_or_exit = input(f"\nIf you are ready, type {bcolors.OKGREEN}start{bcolors.ENDC}.\n"
                              f"If you want to quit, type {bcolors.FAIL}exit{bcolors.ENDC}.\n"
                              f"Output in file %s.\n" 
                              f"Your input: "% outputfile)
        if start_or_exit == 'exit':
            sys.exit(0)
        if start_or_exit == 'start':
            print(f'{bcolors.OKGREEN}-{bcolors.ENDC}'*(2*size_of_each_dialog_line+space_between_dialogs))
            break

    exit_typed = False


    # start comparisons
    if not continue_old_file:
        annotated_data = {}
        annotated_data['data'] = []
        annotated_data['counter'] = 0

    for i in range(starting_sample,len(parsed_comparisons)):  # create n comparisons
        print("This is the dialog history:\n"+'-'*(2*size_of_each_dialog_line+space_between_dialogs))

        query = parsed_comparisons[i]["Q"]
        left_response = parsed_comparisons[i]["S0"]
        right_response = parsed_comparisons[i]["S1"]

        col_query = format_context(query)

        state_str = ("[Sample {current} of {all}]").format(current=str(i+1) , all=str(len(parsed_comparisons)))
        print(f"{bcolors.TEXT}%s{bcolors.ENDC}" % state_str)
        print((col_query.replace('<SOC>','\n'))+'\n'+'-'*(2*size_of_each_dialog_line+space_between_dialogs))
        print('\nWhich next turn is better?\n' + '-'*(2*size_of_each_dialog_line+space_between_dialogs))
        print_dialogs_side_by_side(left_response, right_response,
                                    size=size_of_each_dialog_line,
                                    space=space_between_dialogs)
        print('-'*(2*size_of_each_dialog_line+space_between_dialogs))

        annotator_decision = None

        # start timer
        start_time = time.time()
        while True:
            annotator_decision = input("\nWhich dialog is better? "
                                        f"Type {bcolors.OKGREEN}1{bcolors.ENDC} for left, "
                                        f"{bcolors.OKGREEN}2{bcolors.ENDC} for right, "
                                        f"{bcolors.OKGREEN}3{bcolors.ENDC} if they are equally good, "
                                        f"{bcolors.OKGREEN}4{bcolors.ENDC} if they are uncomparable "
                                        f"(or {bcolors.FAIL}exit{bcolors.ENDC} to stop annotation) and hit enter.\n"
                                        f"Your input: ")

            if annotator_decision == 'exit':
                exit_typed = True
                break

            if annotator_decision in ['1', '2', '3', '4']:
                decision_duration = '%.2f' % (time.time() - start_time)

                # replace speaker information with start of line characters in both dialogs and
                # remove csv delimiter character '|' to prevent bugs
                left_dialog_in_one_line = left_response.replace('\n[A]: ', '<SOL>').replace('\n[B]: ',' <SOL>').replace('|', '')
                right_dialog_in_one_line = right_response.replace('\n[A]: ', '<SOL>').replace('\n[B]: ', '<SOL>').replace('|', '')
                
                annotated_data['data'].append({
                    'query': query,
                    'sample0': left_dialog_in_one_line,
                    'sample1': right_dialog_in_one_line,
                    'annotator_decision': int(annotator_decision),
                    'decision_duration': decision_duration
                })
                # save number of samples you already annotated
                annotated_data['counter'] = i+1
                with open(results_filepath, 'w', encoding='utf-8') as outfile:
                    json.dump(annotated_data, outfile)
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
              % str(results_filepath),)

if __name__ == '__main__':
    annotate()

