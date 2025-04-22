import pandas as pd
import os
import glob


#models = ["h4b0s1kf_buzz", "25yjsq7a_echo", "bki984uw_echo"] #["ag7twbuq_barks", "g8gtuypk_whistle", "h4b0s1kf_buzz", "25yjsq7a_echo"] # ["l4mm3x56", "kfwviyy8", "zyle4p49"] # barks, whis, buzz
models =["barks_4af2w6lt","bki984uw_echo","ccgojzau_buzz","g8gtuypk_whistle"]

ths = [0.5, 0.9, 0.8, 0.5]

for (model,th) in zip(models,ths):
    os.system(f'sh run_inference_all.sh {model} {th}')
