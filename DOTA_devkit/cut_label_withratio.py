import dota_utils as util
import random
import pdb

labelparent='/workspace/ReDet/datasets/HRSC2016/Train/labelTxt_0.1'
filenames = util.GetFileFromThisRootDir(labelparent)
print(len(filenames))

filenames_clear=random.sample(filenames, 617-int(617*0.1))
print(len(filenames_clear))

for file in filenames_clear:
    #clear
    with open(file, 'r+') as f:
        f.truncate(0)