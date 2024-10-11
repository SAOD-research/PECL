'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-06-06 15:06:37
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-06-06 15:12:42
FilePath: /ReDet/DOTA_devkit/deletenoobjectimg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import dota_utils as util
import os
import cv2
import json
from PIL import Image
import pdb

def list_txt(path, list=None):
    '''

    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

aa=list_txt('/workspace/ReDet/datasets/DOTA_1024/trainval1024/delete_file.txt')
print(aa)
print(len(aa))
labelparent='/workspace/ReDet/datasets/DOTA_1024/trainval1024/labelTxt_0.05_mani'
filenames = util.GetFileFromThisRootDir(labelparent)
i=0
for file in filenames:
    basename = util.custombasename(file)
    if basename in aa:
        print(basename)
        # pdb.set_trace()
        os.remove('/workspace/ReDet/datasets/DOTA_1024/trainval1024/labelTxt_0.05_mani/'+basename+'.txt')
        i=i+1
print(i)
