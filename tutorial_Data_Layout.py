#Everything we need IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import librosa
import mir_eval
import mir_eval.display
import tables
import IPython.display
import os
import json

#Local path constants
DATA_PATH = 'data'
RESULTS_PATH = 'results'

#Path to the file match_scores.json  distributed with the LMD
SCORE_FILE = os.path.join(RESULTS_PATH, 'match_scores.json')

#Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefixself.
    E.g TRABCD123456789 -> A/B/C/TRABCD123456789"""
    return os.path.join(msd_id[2],msd_id[3],msd_id[4], msd_id)

def msd_id_to_mp3(msd_id):
    """Given an MSD ID, return the path to the corresponding mp3"""
    return os.path.join(DATA_PATH,'msd','mp3', msd_id_to_dirs(msd_id)+'.mp3')

def msd_id_to_h5(h5):
    """Given an MSD ID,return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH,'lmd_matched_h5',msd_id_to_dirs(msd_id)+'.mp3')

def get_midi_path(msd_id, midi_md5, kind):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind shoud be one of 'matched' or 'aligned'-"""
    return os.path.join(RESULTS_PATH,'lmd_{}'.format(kind), msd_id_to_dirs(msd_id),midi_md5+'.mid')

#DATA LAYOUT
with open(SCORE_FILE) as f:
    scores = json.load(f)
#Takes the Million Song Dataset ID from the scores dictionary

#has to be change to work on python3 adding list and the [] is on the outside
#teh value on the[] is the position of the ID on the table
msd_id = list(scores.keys())[1234]
print('Million Song Dataset ID {} has {} MIDI file matches:'.format(
    msd_id, len(scores[msd_id])))
for midi_md5, score in scores[msd_id].items():
    print ('   {} with confidence score{}'.format(midi_md5, score))
