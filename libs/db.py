import wfdb
import soundfile
import numpy

MAXBLOCKSIZE = 2**21

def rec_id_to_string(rec_id):
    return str(rec_id[0]) + '_' + str(rec_id[1]).zfill(4)

def rec_id_to_tuple(rec_id):
    return int(rec_id.split('_')[0]), int(rec_id.split('_')[1])

def to_digital(data, hdr):
    data *= hdr.adc_gain
    data += hdr.baseline
    data[numpy.isnan(data)] = -2**15
    data = numpy.around(data).astype('int16')
    return data

def to_physical(data, hdr):
    data = data.astype('float64')
    data[data==-2**15] = numpy.nan
    data -= hdr.baseline
    data /= hdr.adc_gain
    return data

def sig_is_y(sig_name):
    y_names = wfdb.io.SIGNAL_CLASSES.to_dict()['signal_names']['bp']
    return sig_name.lower() in y_names

def read_waveforms(hdr, path):
    read_args = {'dtype': 'int16', 'always_2d': True}
    if hdr.sig_len < MAXBLOCKSIZE:
        data, rate = soundfile.read(path + '.flac', **read_args)
    else:
        data = numpy.empty((hdr.sig_len, hdr.n_sig), dtype='int16')
        with soundfile.SoundFile(path + '.flac') as sf:
            blocks = sf.blocks(blocksize=MAXBLOCKSIZE, **read_args)
            for i, block in enumerate(blocks):
                data[i*MAXBLOCKSIZE:(i+1)*MAXBLOCKSIZE] = block
    return data

def read_original_record(rec_id, compute_physical=True):
    if type(rec_id) is tuple:
        rec_id = rec_id_to_string(rec_id)
    path = 'mimic3wdb/' + rec_id[:2] + '/' + rec_id.split('_')[0]
    rec = wfdb.rdrecord(rec_id, pb_dir=path, physical=compute_physical)
    return rec

def read_record(rec_id, root='/scr1/mimic/waveforms/', x_only=False, y_only=False, compute_physical=True):

    if type(rec_id) is tuple:
        rec_id = rec_id_to_string(rec_id)
        
    path = root + rec_id
    hdr = wfdb.rdheader(path)
    is_y = numpy.array([sig_is_y(j) for j in hdr.sig_name])

    if x_only or (~is_y).all():
        rec = wfdb.rdheader(path + '_x')
        data = read_waveforms(rec, path + '_x')
    elif y_only or is_y.all():
        rec = wfdb.rdheader(path + '_y')
        data = read_waveforms(rec, path + '_y')
    else:
        rec = hdr
        rec_x = wfdb.rdheader(path + '_x')
        rec_y = wfdb.rdheader(path + '_y')
        data = numpy.empty((rec.sig_len, rec.n_sig), dtype='int16')
        data[:, ~is_y] = read_waveforms(rec_x, path + '_x')
        data[:,  is_y] = read_waveforms(rec_y, path + '_y')

    if compute_physical:
        rec.p_signal = to_physical(data, rec)
    else:
        rec.d_signal = data

    return rec

def write_record(rec, path):
    data = to_digital(rec.p_signal.copy(), rec)
    with open(path + '.flac', 'wb') as f:
        soundfile.write(f, data, 125, format='FLAC')
