import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from pathlib import Path
import re
import os
import json
from datetime import datetime,timedelta


def df_freq_range_dict(x):
    return {
        'LF-200Hz': 'HYDBBA105_Calls_2018_04_200Hz.csv',
        'LF-2kHz': 'HYDBBA105_Calls_2018_04_2kHz.csv',
        'MF': 'HYDBBA105_Calls_2018_04_EvenDays_MF.csv'

    }.get(x, None)

def df_attach_wav_files(sub_df, row):
    folder_prefix = '2018_04/OOI_105_'
    file_prefix = 'HYDBBA105OOI_'
#     date_time_obj = datetime. strptime(row['UTC'], '%Y-%m-%d %H:%M:%S.%f')
    format_1 = '%m/%d/%Y %H:%M:%S'
    format_2 = '%m/%d/%Y %H:%M'
    try:
        date_time_obj = datetime. strptime(row['UTC'], format_1)
    except:
        try:
            date_time_obj = datetime. strptime(row['UTC'], format_2)
        except ValueError:
            print("problematic time format!")


    date_time_obj = date_time_obj - timedelta(minutes=date_time_obj.minute % 5,
                                seconds=date_time_obj.second,
                                microseconds=date_time_obj.microsecond)
#     '4/1/2018 12:05:53'
    begin_time = date_time_obj - timedelta(minutes=(date_time_obj.minute - (date_time_obj.minute % 5)),
                                seconds=date_time_obj.second,
                                microseconds=date_time_obj.microsecond)
    file_address = datetime.strftime(date_time_obj,folder_prefix + '%Y_%m_%d/' + file_prefix +'%Y%m%d-%H%M%S.wav')

    return file_address
    

def process_row(row, metadata_dict):
#     print(row['PG_UID'])
    freq_range_df = df_freq_range_dict(row['Run'])
    if freq_range_df is None:
        print('No such df!')
    else:
        chosen_freq_df = metadata_dict[freq_range_df]
        sub_df = chosen_freq_df[chosen_freq_df['parentUID'] == row['PG_UID']]
        sub_df = sub_df[['UID','UTC','UTCMilliseconds']]
        sub_df['Species_ID'] = row['Species_ID']
        sub_df['Sound_type'] = row['Sound_type']
        sub_df['PG_UID'] = row['PG_UID'].copy()
        sub_df['file'] = df_attach_wav_files(sub_df, row)
        
    
    return sub_df

def main():
    print("Hello World!")
    selection_path = Path('../..//multispecies_dataset')
    filelist = list(selection_path.glob('**/*.csv'))

    metadata =[]
    metadata_dict = {}
    for i,file in enumerate(filelist):
        dfTemp = pd.read_csv(file)
        print(file.name)
        print('Columns names')
        print(dfTemp.columns)
        print('Dataframe shape')
        print(dfTemp.shape)
    #     dfTemp['filename'] = re.search(r'18\d{4,4}_\d{6,6}', file.as_posix()).group() 
    #     dfTemp['StartMicInWater'] =np.amin(dfTemp['Begin Time (s)'])
    #     dfTemp['EndMicInWater'] =np.amax(dfTemp['End Time (s)'])
        metadata.append(dfTemp)
        metadata_dict[file.name] = dfTemp
    Calls_2018_04_2kHz, Calls_2018_04_EvenDays_MF, Calls_2018_04_200Hz,EvenDays_AllRuns_v2 = metadata


    chosen_categories = ['Delph sp.   ', 'Fin whale   ', 'Mn          ']
    filtered_categories_df = EvenDays_AllRuns_v2[EvenDays_AllRuns_v2['Species_ID'].isin(chosen_categories)]
    total_df = []
    for _, row in filtered_categories_df.iterrows():
    #     print(row)
        total_df.append(process_row(row, metadata_dict))

    total_df = pd.concat(total_df)
    print('finished parsing dataset')


if __name__ == "__main__":
    main()

