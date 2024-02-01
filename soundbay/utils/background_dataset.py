# This script takes the txt files with "txt_BG" (tab separated) in a certain directory and creates a dataframe of the data. 
# Then it sums the values of the column "Duration" and prints it as the total amount of background annotation in seconds.

import pandas as pd
import os


# Path to the directory containing the txt files
path = r"C:/Users/amitg/Documents/Deep_Voice/ocean-whispers/Anotations Ocean Wispers/"

# Create a list of all the txt files in the directory with "txt_BG" in their name
files_BG = [f for f in os.listdir(path) if "txt_BG" in f]

# Create an empty dataframe
df = pd.DataFrame()

# Iterate over the files and append them to the dataframe, create a column "Duration" with the difference between the columns "Begin Time (s)" and "End Time (s)"
for f in files_BG:
    data = pd.read_csv(path + f, sep="\t")
    df = df._append(data)
    df["Duration"] = df["End Time (s)"] - df["Begin Time (s)"]

print(df.head())
# Sum the values of the column "Duration" and print it as the total amount of background annotation in seconds
print("Amount of BG:", df["Duration"].sum())

# Now doing the same for the non-background annotations, for all txt files with no "txt_BG" in their name
files = [f for f in os.listdir(path) if "txt_BG" not in f]

df2 = pd.DataFrame()

for f in files:
    data = pd.read_csv(path + f, sep="\t")
    df2 = df2._append(data)
    df2["Duration"] = df2["End Time (s)"] - df2["Begin Time (s)"]

print(df2.head())
print("Amount of non-BG:", df2["Duration"].sum())

