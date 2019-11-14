#!/usr/bin/env python
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
sys.path.append('/home/lichao/projects/pytracking_lichao')
from pytracking.evaluation import Tracker

def run_vot(imgtype):
    
    tracker = Tracker('etcom_comb2', 'debug_lichao_comb23_2')
    tracker.run_vot2(imgtype)

run_vot('ir')

