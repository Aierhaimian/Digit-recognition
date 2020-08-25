#!/usr/bin/python3
#-*- coding: utf-8 -*-

import os

def rename_func(prefix, path):
    cnt = 0
    for file in os.listdir(path):
        fileNew = prefix + str(cnt) + '.jpg'
        cnt += 1
        os.rename(path + file, path + fileNew)


if __name__ == "__main__":
    prefix = '0_'
    path = '../dataset/0/'
    rename_func(prefix, path)