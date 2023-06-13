import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import glob
import re

if __name__ == "__main__":
    patt = "perps-cf-diff-sizes/20000000/1/*-REAL_REAL_full_results.csv"

    for filename in glob.glob(patt):
        lang = re.search("..(?=-REAL_REAL)", filename)
        assert lang
        lang = lang.group(0)
        df = pd.read_csv(filename)
        sum_surp = df.surprisal.sum()
        sum_charlens = df.token.asstype(str).apply(len).sum()
        bits_per_char = sum_surp / sum_charlens
        print(f"Lang: {lang}, Bits per Char: {bits_per_char}")
