import pandas as pd
import sys

file = sys.argv[1]
df = pd.read_csv(file)
df = df.sort_values(by=['filename', 'begin_time'])
for (i,n) in enumerate(["device", "day", "hour"]):
     df[n]=df['filename'].str.split('/').str[-1].str.split('_').str[i]
df['hour'] = df['hour'].str.replace('.WAV', '')
df.to_csv(file, index=False)
