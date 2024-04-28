#!/bin/bash

# Iterate over files
for file in *d.csv; do
    # Remove the 'd' at the end of the filename
    base_filename="${file%d.csv}"

    # Check if the corresponding file without 'd' exists
    if [ -e "${base_filename}.csv" ]; then
        # Run diff if the corresponding file exists
        if ! diff -q "${base_filename}.csv" "$file" >/dev/null; then
            echo "Files ${base_filename}.csv and $file differ."
            echo "Differences:"
            diff "${base_filename}.csv" "$file" | grep -E '^<|^>'
        fi
    else
        echo "Corresponding file ${base_filename}.csv not found."
    fi
done
