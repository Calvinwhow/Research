#!/bin/bash
for dir in /data/*; do
    sub=$(basename "$dir")
    echo "Found dir: $sub"
done