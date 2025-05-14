#!/bin/bash

for file in dataset/*.txt; do
    python3 main.py --filename "$file"
done