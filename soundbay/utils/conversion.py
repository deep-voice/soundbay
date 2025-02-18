import pandas as pd
import os
from pathlib import Path

directory = "/Users/shai/personal/deepvoice/amber/results_oct26/raven_combined_test1/"

def _convert(directory):

    # Process each .csv file
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Construct full file paths
            csv_path = os.path.join(directory, filename)
            txt_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.txt")

            # Load the CSV file
            df = pd.read_csv(csv_path,
                             sep="\t",
                             quotechar='"',   # Handle quoted fields
                             skipinitialspace=True)

            # Save as TXT with tab-separated values
            df.to_csv(txt_path, sep="\t", index=False)

    print("Conversion complete!")


def _change_high_freq_for_buzz(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Construct full file path
            txt_path = os.path.join(directory, filename)

            # Load the TXT file as a DataFrame (assuming tab-separated values)
            df = pd.read_csv(txt_path,
                             sep="\t",
                             quotechar='"',   # Handle quoted fields
                             skipinitialspace=True)

            # Check if the required columns exist
            if 'Class Name' in df.columns and 'High Freq (Hz)' in df.columns:
                # Update 'High Freq' where 'Class' is 'buzz'
                df.loc[df['Class Name'] == 'buzz', 'High Freq (Hz)'] = 48000

                # Save the updated DataFrame back to the same file
                df.to_csv(txt_path, sep="\t", index=False)
            else:
                print(f"Skipped {filename}: Required columns not found.")

    print("Processing complete!")


def fix_bug(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Read the file as a single column
            df = pd.read_csv(directory + filename,
                        sep="\t",
                        quotechar='"',  # Handle quoted fields
                        skipinitialspace=True,
                        header=None)

            # Split the first column into separate columns
            if len(df) > 1:
                df_split = df[0].str.split("\t", expand=True)

                # Rename the columns (use the header from the first row or define custom names)
                df_split.columns = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
                                    "Low Freq (Hz)", "High Freq (Hz)", "Annotation", "Class Name", "Probability"]

                df_split = df_split.iloc[1:]
                # Ensure the "Selection" column is an integer type
                df_split["Selection"] = range(1, len(df_split) + 1)

                # Convert the "Selection" column to integer explicitly (if necessary)
                df_split["Selection"] = df_split["Selection"].astype(int)
                df_split.to_csv(directory + Path(filename).stem + ".txt", sep="\t", index=False)


if __name__ == "__main__":
    # _change_high_freq_for_buzz(directory)
    # _convert(directory)
    fix_bug(directory)
