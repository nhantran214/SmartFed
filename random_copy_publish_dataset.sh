#!/bin/bash

# Check for input and output paths as arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/input_dir /path/to/output_dir"
    exit 1
fi

input_dir=$1
output_dir=$2

# Create a temporary file to store the randomly selected files
temp_file=$(mktemp)

# Find all files in the input directory, shuffle them, and select 20,000 randomly
find "$input_dir" -type f | shuf -n 1000 > "$temp_file"

# Copy each randomly selected file to the output directory
while IFS='' read -r file || [[ -n "$file" ]]; do
    cp "$file" "$output_dir"
done < "$temp_file"

# Clean up by removing the temporary file
rm "$temp_file"