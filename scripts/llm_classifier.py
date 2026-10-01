import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import import_json_data

import_json_data("deepseek-r1:32b_Nonesamples_verbose_full1_swappingFalse.json")