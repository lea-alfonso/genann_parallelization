#!/usr/bin/env python

import pandas as pd
import zipfile
import os
 
# assign directory
directory = '/home/leandro/Downloads/raw_compressed_files/'
directory_to_extract_to ="/home/leandro/Downloads/raw_extracted_files"
directory_to_save_when_finished = "/home/leandro/Downloads/reorganized_csv_files"
#directory_to_compress_them = "/home/leandro/Downloads/parquets"
 
#for filename in os.scandir(directory):
#    if filename.is_file() and os.path.splitext(filename)[1] == ".zip":
#        with zipfile.ZipFile(filename.path, 'r') as zip_ref:
#            print(f"EXTRACTING {filename}")
#            zip_ref.extractall(directory_to_extract_to)

for filename in os.scandir(directory_to_extract_to):
    if filename.is_file() and os.path.splitext(filename)[1] == ".csv":
        print(f"\t READING {filename}")
        encode=None
        try:
            df = pd.read_csv(filename.path,dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(filename.path,encoding="latin_1",dtype=str)
            encode = "latin_1"

        if len(df.columns) == 1 :
            df = pd.read_csv(filename.path,sep=";",encoding=encode,dtype=str)

        if "id_detector" not in df.columns and "cod_detector" in df.columns :
            print("\t RENAMING")
            df.rename(columns={"cod_detector":"id_detector","volume": "volumen"},inplace=True)
        elif "id_detector" not in df.columns:
            print("\t I DONT KNOW WHAT TO DO")
            print(df.dtypes)
        print(f"\t Appending {filename}")
        for id_detector in df["id_detector"].unique():
            df_id = df.loc[df["id_detector"] == id_detector]
            output_path = os.path.join(directory_to_save_when_finished,"id_detector_"+ id_detector + ".csv")
            df_id.to_csv(output_path,mode="a",header= not os.path.exists(output_path),index=False)
            #os.remove(filename.path)

#for filename in os.scandir(directory_to_save_when_finished):
#    if filename.is_file() and os.path.splitext(filename)[1] == ".csv":
#        df = pd.read_csv(filename.path,encoding="latin_1")
#        df.to_parquet(directory_to_compress_them + os.path.splitext(filename)[0] + ".parquet")



