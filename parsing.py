##### SCRIPT STARTS HERE #####
#!usr/bin/bash python

import argparse

parser = argparse.ArgumentParser(description='Trains a OVCA Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--histotypes', action='store', type=str, nargs="+", 
                    help='space separated str IDs specifying histotype labels')


args = parser.parse_args()
print(args.histotypes)
