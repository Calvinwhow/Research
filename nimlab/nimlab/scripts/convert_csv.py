#!/bin/python3
from nimlab.nimtrack import *
import nimlab.datasets as nimds
import pandas as pd
import csv
import argparse
from bids import layout
import json

parser = argparse.ArgumentParser(description='Convert xnat filepaths to nimtrack filepath')
parser.add_argument('xnat_csv', type=str, help='input csv')
parser.add_argument('output_csv', type=str, help='output file name')
parser.add_argument('--prefix', type=str, help='Common filepath prefix', default='/data/nimlab/xnat_archive',)

args = parser.parse_args()

with open(nimds.get_filepath('connectivity_config')) as f:
    config = json.load(f)

patterns = config['default_path_patterns']
with open(args.xnat_csv) as csvfile:
    reader = csv.reader(csvfile)
    with open(args.output_csv, 'w') as writefile:
        writer = csv.writer(writefile)
        for row in reader:
            newrow = []
            for item in row:
                if("/xnat_archive/" in item):
                    converted = convert_xnat_path(item, "/data/nimlab/dl_archive/",args.prefix)
                    newrow.append(converted)
                    #print(converted)
                else:
                    # Paths not within xnat archive will not be affected
                    newrow.append(item)
            writer.writerow(newrow)

