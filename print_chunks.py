#!/usr/bin/env python3
import os
from config import CONFIG
from data_loader import load_data

script_dir = os.path.dirname(os.path.abspath(__file__))
data = load_data(script_dir, CONFIG)
all_combined_texts = data["all_combined_texts"]

for i, chunk in enumerate(all_combined_texts):
    print(f"Chunk {i}: {chunk[:200]}...")  # print first 200 chars