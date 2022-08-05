from tqdm import tqdm
import pandas as pd
from pathlib import Path
from datetime import datetime,timedelta
from soundbay.utils.metadata_processing import non_overlap_df

def df_freq_range_dict(x):
    return {
        'LF-200Hz': 'HYDBBA105_Calls_2018_04_200Hz.csv',
        'LF-2kHz': 'HYDBBA105_Calls_2018_04_2kHz.csv',
        'MF': 'HYDBBA105_Calls_2018_04_EvenDays_MF.csv'

    }.get(x, None)

def df_attach_wav_files(row):
    folder_prefix = '2018_04/OOI_105_'
    file_prefix = 'HYDBBA105OOI_'
#     date_time_obj = datetime. strptime(row['UTC'], '%Y-%m-%d %H:%M:%S.%f')
    format_1 = '%m/%d/%Y %H:%M:%S'
    format_2 = '%m/%d/%Y %H:%M'
    format_3 = '%Y-%m-%d %H:%M:%S.%f'
    try:
        date_time_obj_orig = datetime.strptime(row['UTC'], format_1) 
        if 'UTCMilliseconds' in row.index:
            date_time_obj_orig = date_time_obj_orig + timedelta(milliseconds=row['UTCMilliseconds'])
    except:
        try:
            date_time_obj_orig = datetime.strptime(row['UTC'], format_2)
            if 'UTCMilliseconds' in row.index:
                date_time_obj_orig = date_time_obj_orig + timedelta(milliseconds=row['UTCMilliseconds'])
        except:
            try:
                date_time_obj_orig = datetime.strptime(row['UTC'], format_3)
                # if 'UTCMilliseconds' in row.index:
                #     date_time_obj_orig = date_time_obj_orig + timedelta(milliseconds=row['UTCMilliseconds'])
            except ValueError:
                print("problematic time format!")
                return None, None, None, None



    date_time_obj_for_wav_file = date_time_obj_orig - timedelta(minutes=date_time_obj_orig.minute % 5,
                                seconds=date_time_obj_orig.second,
                                microseconds=date_time_obj_orig.microsecond)
#     '4/1/2018 12:05:53'

    time_interval = timedelta(microseconds=500000) # Half second interval
    begin_time = (date_time_obj_orig - date_time_obj_for_wav_file - time_interval).total_seconds()

    end_time = (date_time_obj_orig - date_time_obj_for_wav_file + time_interval).total_seconds()

    if begin_time < 0 or end_time > 5 * 60:
        return None, None, None, None

    call_length = end_time - begin_time

    file_address = datetime.strftime(date_time_obj_for_wav_file,folder_prefix + '%Y_%m_%d/' + file_prefix +'%Y%m%d-%H%M%S.wav')

    
    return call_length, begin_time, end_time, file_address
    

def process_row(row, metadata_dict):
#     print(row['PG_UID'])
    freq_range_df = df_freq_range_dict(row['Run'])
    if freq_range_df is None:
        print('No such df!')
    else:
        chosen_freq_df = metadata_dict[freq_range_df]
        sub_df = chosen_freq_df[chosen_freq_df['parentUID'] == row['PG_UID']]
        sub_df = sub_df.reset_index()
        sub_rows = []
        for ind,sub_row in tqdm(sub_df.iterrows(), desc='iterating_rows'):
            # sub_df = sub_df[['UID','UTC','UTCMilliseconds']]
            # sub_row['Species_ID'] = row['Species_ID']
            # sub_row['Sound_type'] = row['Sound_type']
            # sub_row['PG_UID'] = row['PG_UID']
            call_length, begin_time, end_time, file_address = df_attach_wav_files(sub_row)

            if call_length is None:
                continue
            d = {'Species_ID': [row['Species_ID']], 
            'Sound_type': [row['Sound_type']],
            'PG_UID': [row['PG_UID']],
            'call_length': [call_length],
            'begin_time': [begin_time],
            'end_time': [end_time],
            'call_length': [call_length],
            'file_address': [file_address],
            'freq_range': [row['Run']],
            'UID':[sub_row['UID']]}

            df = pd.DataFrame(data=d)


            sub_rows.append(df)
    sub_rows = pd.concat(sub_rows)
    # sub_rows = pd.DataFrame(sub_rows)
    return sub_rows



def main():
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
        metadata.append(dfTemp)
        metadata_dict[file.name] = dfTemp
    Calls_2018_04_2kHz, Calls_2018_04_EvenDays_MF, Calls_2018_04_200Hz,EvenDays_AllRuns_v2 = metadata


    chosen_categories = ['Delph sp.   ', 'Fin whale   ', 'Mn          ', 'Pm          ']
    filtered_categories_df = EvenDays_AllRuns_v2[EvenDays_AllRuns_v2['Species_ID'].isin(chosen_categories)]
    total_df = []
    for _, row in filtered_categories_df.iterrows():
        total_df.append(process_row(row, metadata_dict))

    total_df = pd.concat(total_df)
    total_df.to_csv('pre_merge.csv', index=False)
    total_df = pd.read_csv('./pre_merge.csv')
    print(f'previous sizes of dataframe: {total_df.shape}')
    total_df.rename(columns={'file_address':'filename'}, inplace=True)
    total_df_merged = non_overlap_df(total_df)
    print(f'previous sizes of dataframe: {total_df_merged.shape}')
    total_df_merged.to_csv('post_merge.csv', index=False)

    print('finished parsing dataset')


if __name__ == "__main__":
    main()