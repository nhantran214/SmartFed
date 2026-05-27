import os
import sys

def remove_empty_files(folder_path):
    # Check if the specified path is a directory
    if not os.path.isdir(folder_path):
        print(f"{folder_path} is not a valid directory.")
        return

    # Count for reporting purposes
    removed_files_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file size is 0 bytes and remove it
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Removed: {file_path}")
            removed_files_count += 1

    # Report the result
    if removed_files_count == 0:
        print("No empty files were found.")
    else:
        print(f"Removed {removed_files_count} empty file(s).")

if __name__ == "__main__":
    # Check if the folder path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python remove_empty_files.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    remove_empty_files(folder_path)