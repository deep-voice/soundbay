# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:42:57 2018

@author: kristenk
"""

from onc.onc import ONC
import csv
import datetime

#Specify user access token - I have my token below
token='INSERT-YOUR-TOKEN-HERE'

#Specify outPath and path to Download List  [update to the proper path on your computer]
outPath='downloaded_dataset'
fn="DownloadListpy-filtered_orca.csv"

def format_time(dt):
    return "%s:%06.3f%sZ" % (
            dt.strftime("%Y-%m-%dT%H:%M"),
            float("%.3f" % (dt.second+dt.microsecond /1e6)),
            dt.strftime('%z'))


#GET START,STOP, AND FILENAMES FOR EACH FILE
with open(fn,"r") as f:
    reader=csv.reader(f)
    for row in reader:
        fnamewav,start=row
        try:
            startdatetime = datetime.datetime.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ')
            stopdatetime = startdatetime+datetime.timedelta(seconds=1)
            stop=format_time(stopdatetime)


            production=True
            showInfo=False


            #Create object for subsequent calls
            onc=ONC(token,production,showInfo,outPath)

            maxRetries=20
            downloadResultsOnly=False
            includeMetadataFile=True


            orders = onc.orderDataProduct( { 'locationCode' : 'BACUS',

                    'deviceCategoryCode' : 'HYDROPHONE',

                    'dataProductCode' : 'AD',

                    'extension' : 'wav',

                    'dateFrom' : start,

                    'dateTo' : stop,

                    'dpo_hydrophoneDataDiversionMode' : 'All',

                    'dpo_audioDownsample': -1,

                    'dpo_audioFormatConversion': 0,

                    },

                                maxRetries, downloadResultsOnly, includeMetadataFile )
        except Exception:
            print(f'Failed on file: {fnamewav}')
            pass
