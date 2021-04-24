import base64
import requests
import json
import os
import csv
import re
import random
import pandas as pd

if __name__ == '__main__':
    path = "sample_batch_results.csv"
    df = pd.read_csv(path, index_col=False)
    real_A = df["Input.sense_img_url"]
    img1 = df["Input.input_img1_url"]
    img2 = df["Input.input_img2_url"]
    selected_label = df['Answer.category.label']

