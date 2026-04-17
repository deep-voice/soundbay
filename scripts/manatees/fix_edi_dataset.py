import pandas as pd
import os

# Load the CSV file
csv_path = r'C:\Users\amitg\OneDrive\Documents\Deep_Voice\HF_WAV_Manatee_Samples\edi.2108.2\edi_manatee_with_Ntem_3Ipono2_16k.csv'
df = pd.read_csv(csv_path)

# Display the head of the 'filename' column to confirm its structure
print("Original 'filename' column head:")
print(df['filename'].head())

# Define a function to fix the filename
# The current format is "subfolder/stem" (e.g., "Noise/20220324T...")
# We want just the "stem" (e.g., "20220324T...")
def fix_filename(path_str):
    if '/' in path_str:
        return path_str.split('/')[-1]
    return path_str

# Apply the function to the 'filename' column
df['filename'] = df['filename'].apply(fix_filename)

# Display the head of the new 'filename' column to show the fix
print("\nFixed 'filename' column head:")
print(df['filename'].head())

# Save the corrected DataFrame to a new CSV file
new_csv_path = 'edi_manatee_metadata_fixed.csv'
df.to_csv(new_csv_path, index=False)

print(f"\nSuccessfully saved corrected metadata to: {new_csv_path}")
