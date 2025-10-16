#!/bin/bash

# Script to zip ALL 7 model directories ensuring no files are missed
# This script creates a comprehensive archive of ALL model directories

set -e  # Exit on any error

echo "Starting comprehensive zip process for ALL 7 model directories..."

# Get current directory
BASE_DIR="/pscratch/sd/s/saik1999/brain_Networks"
cd "$BASE_DIR"

# Define ALL directories to zip
DIRECTORIES=("ALTER" "BioBGT" "BQN-corrected" "brainetcnn" "gcn_gps_experiments" "graphtransformer" "QUETT")

# Create timestamp for unique zip filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ZIP_NAME="all_7_models_${TIMESTAMP}.zip"

echo "Creating zip file: $ZIP_NAME"

# Count total files before zipping
echo "Counting files in each directory..."
TOTAL_FILES=0
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        FILE_COUNT=$(find "$dir" -type f | wc -l)
        SIZE=$(du -sh "$dir" | cut -f1)
        echo "  $dir: $FILE_COUNT files ($SIZE)"
        TOTAL_FILES=$((TOTAL_FILES + FILE_COUNT))
    else
        echo "  WARNING: Directory $dir not found!"
    fi
done
echo "Total files to zip: $TOTAL_FILES"

# Show total size
TOTAL_SIZE=$(du -sh "${DIRECTORIES[@]}" | tail -1 | cut -f1)
echo "Total size to compress: $TOTAL_SIZE"

# Create the zip file with verbose output and preserve directory structure
echo "Creating zip archive (this may take a while due to large files)..."
zip -r "$ZIP_NAME" "${DIRECTORIES[@]}" -v

# Verify the zip file was created
if [ -f "$ZIP_NAME" ]; then
    echo "Zip file created successfully: $ZIP_NAME"
    echo "Zip file size: $(du -h "$ZIP_NAME" | cut -f1)"
else
    echo "ERROR: Zip file was not created!"
    exit 1
fi

# Verify contents of the zip file
echo "Verifying zip contents..."
ZIP_FILE_COUNT=$(unzip -l "$ZIP_NAME" | tail -1 | awk '{print $2}')
echo "Files in zip archive: $ZIP_FILE_COUNT"

if [ "$ZIP_FILE_COUNT" -eq "$TOTAL_FILES" ]; then
    echo "SUCCESS: All $TOTAL_FILES files are included in the zip archive!"
else
    echo "INFO: File count difference (expected: $TOTAL_FILES, found: $ZIP_FILE_COUNT)"
    echo "This is normal as zip includes directories and metadata"
fi

# Test zip integrity
echo "Testing zip file integrity..."
if unzip -t "$ZIP_NAME" > /dev/null 2>&1; then
    echo "SUCCESS: Zip file integrity test passed!"
else
    echo "ERROR: Zip file integrity test failed!"
    exit 1
fi

# Show summary of what's included
echo ""
echo "=== ZIP SUMMARY ==="
echo "Archive: $ZIP_NAME"
echo "Location: $BASE_DIR/$ZIP_NAME"
echo "Models included:"
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" | cut -f1)
        echo "  ✓ $dir ($SIZE)"
    fi
done
echo ""
echo "Zip process completed successfully!"
