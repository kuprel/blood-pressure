import os
import numpy
import wfdb
import flacdb
import bloodpressure
import shutil
from multiprocessing import Pool, cpu_count

def write_part(rec, path, is_part):
    rec_part = wfdb.Record(**rec.__dict__)
    rec_part.p_signal = rec_part.p_signal[:, is_part]
    rec_write_fields, sig_write_fields = rec.get_write_fields()
    for k in sig_write_fields:
        rec_part.__dict__[k] = numpy.array(rec_part.__dict__[k])[is_part].tolist()
    rec_part.n_sig = is_part.sum()
    rec_part.record_name = path.split('/')[-1]
    bloodpressure.write_record(rec_part, path)
    write_dir = '/'.join(path.split('/')[:-1])
    rec_part.wr_header_file(rec_write_fields, sig_write_fields, write_dir)

def split(path):
    rec = flacdb.read_record(path)
    is_y = numpy.array([bloodpressure.sig_is_y(j) for j in rec.sig_name])

    if is_y.all():
        shutil.copy(path + '.flac', path + '_y.flac')
        shutil.copy(path + '.hea', path + '_y.hea')
    else:
        write_part(rec, path + '_x', ~is_y)
        write_part(rec, path + '_y', is_y)

    rec_ = bloodpressure.read_record(path)
    assert(rec == rec_)
    os.remove(path + '.flac')

def _split(path):
    try:
        split(path)
    except:
        print(path)

if __name__ == '__main__':
    root = '/scr-ssd/mimic3wdb/train/'
    dirs = os.listdir(root)
    paths = [
        root + i.split('.')[0] for i in os.listdir(root)
        if '.flac' in i and 'x' not in i and 'y' not in i
    ]
    paths = sorted(paths)

    Pool(cpu_count()).map(_split, paths)
