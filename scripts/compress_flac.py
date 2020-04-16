import wfdb
import flacdb
import os
import gc
import time
import random

def get_records(path_part):
    dirs = os.listdir(path_part)
    paths = []
    for i in dirs:
        files = os.listdir(path_part + i)
        paths += sorted({path_part + i + '/' + j.split('.')[0] for j in files})
    return paths

db = '/scr1/mimic/new_data_raw/3{}/'

paths = []
for part in range(10): paths += get_records(db.format(part))
paths = sorted(paths)
#     random.shuffle(paths)
    
for path in paths:
    isfile = os.path.isfile(path + '.dat')
    if isfile and os.path.getsize(path + '.dat') > 0:
        if paths.index(path) % 100 == 0:
            gc.collect()
        print(paths.index(path), path)
        rec = wfdb.rdrecord(path)
        flacdb.write_record(rec, path)
        try:
            rec_flac = flacdb.read_record_old(path)
            assert(rec == rec_flac)
            os.remove(path + '.dat')
            open(path + '.dat', 'a').close()
            print('converted {}'.format(path))
        except:
            print('failed to convert {}'.format(path))
