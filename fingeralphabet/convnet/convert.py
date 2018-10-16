#!/usr/bin/env python3

import sh
import random
import glob
import tqdm

rand = random.seed(1337)

sh.rm('-r', '-f', 'train2', 'test2') # clear folders 

files = glob.glob('./*/*/*.jpg')

random.shuffle(files)

test_ind = len(files) // 10

test_fns, train_fns = files[:test_ind], files[test_ind:]

for dst, fns in (('train2', train_fns), ('test2', test_fns)):
    sh.mkdir(dst)
    for c in range(29):
        sh.mkdir('{}/{}'.format(dst, c))
    for fn in tqdm.tqdm(fns):
        fn_part = fn.rsplit('/', 1)[1]
        if fn_part.startswith('NOTHING'):
            c = 28
        elif fn_part.startswith('CH'):
            c = 27
        elif fn_part.startswith('SCH'):
            c = 26
        else:
            c = ord(fn_part[0]) - ord('A')
        sh.cp(fn, '{}/{}'.format(dst, c))
