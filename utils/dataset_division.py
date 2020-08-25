#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os


def dataset_division(path, train, valid, test):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            apath = os.path.join(root, dir)
            for root1, dirs1, files1 in os.walk(apath):
                cnt = 0
                for file in files1:
                    name = str(root1) + '/' + str(file) + ' ' + str(dir) + '\n'
                    if cnt < 420:
                        train.write(name)
                    elif 420 <= cnt < 540:
                        valid.write(name)
                    else:
                        test.write(name)
                    cnt += 1


if __name__ == "__main__":
    path = "/home/dzx/workSpace/Digit-recognition/dataset/"
    train = open('../train.txt', 'a')
    valid = open('../valid.txt', 'a')
    test = open('../test.txt', 'a')

    dataset_division(path, train, valid, test)
