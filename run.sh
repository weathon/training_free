#!/bin/bash

for file in finetrainers/training/mochi-1/dataset/*.txt; do
    python3 mochi_main.py --filename "$file";
done