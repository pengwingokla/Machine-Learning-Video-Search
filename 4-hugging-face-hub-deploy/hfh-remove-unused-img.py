import os
import pandas as pd

def remove_unused_images(csv_filename, image_directory):
    # Read the CSV file
    df = pd.read_csv(csv_filename)

    # Get the list of image files mentioned in the 'file_name' column
    image_files_in_csv = df['file_name'].tolist()

    # Get the list of all image files in the directory
    all_image_files = os.listdir(image_directory)

    # Iterate over all image files in the directory
    for image_file in all_image_files:
        # Check if the image file is not in the CSV file
        if image_file not in image_files_in_csv:
            # Construct the path to the image file
            image_path = os.path.join(image_directory, image_file)
            # Remove the image file
            os.remove(image_path)
            print(f"Removed {image_file}")

# Specify the CSV filename and the directory containing the image files
csv_filename = 'D:\HFH-Dataset-YouTube-Video-Search\\npr-frames\\metadata.csv'
image_directory = 'D:\HFH-Dataset-YouTube-Video-Search\\npr-frames'

# Call the function to remove unused images
remove_unused_images(csv_filename, image_directory)
