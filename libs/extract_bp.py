import os
import numpy
import pickle
import wfdb
import bloodpressure
from scipy import signal, interpolate
from multiprocessing import Pool, cpu_count

MAX_HEART_RATE = 250
MIN_PULSE_PRESSURE = 10
MIN_PRESSURE = 20
MAX_BP_DELTA = 15

fs = 125

def get_peak_indices(y):
    di = 60 / MAX_HEART_RATE * fs
    i = signal.find_peaks(y, prominence=MIN_PULSE_PRESSURE, distance=di)[0]
    if i.size < 2: return i
    dy = numpy.abs(numpy.diff(y[i]))
    keep = (dy[:-1] < MAX_BP_DELTA) | (dy[1:] < MAX_BP_DELTA)
    keep = numpy.concatenate(([True], keep, [True]))
    return i[keep]

def interpolator(i, y):
    if i.size == 0: return lambda j: numpy.full(j.size, numpy.nan)
    return interpolate.interp1d(
        i, y,
        kind='linear' if i.size > 1 else 'zero',
        bounds_error=False,
        fill_value=(y[0], y[-1])
    )

def sample(y, i):
    i_sys = get_peak_indices(y)
    i_dia = get_peak_indices(-y)

    f_sys = interpolator(i_sys, y[i_sys])
    f_dia = interpolator(i_dia, y[i_dia])

    z = numpy.transpose(numpy.vstack((f_sys(i), f_dia(i))))

    i_nan = numpy.isnan(y[i])
    z[i_nan] = numpy.nan
    z[numpy.less(y[i], MIN_PRESSURE, where=~i_nan)] = numpy.nan
    z = z.astype('float16')

    return z

def extract_part(path):
    rec = bloodpressure.read_record(path, y_only=True)
    y = rec.p_signal[:, rec.sig_name.index('ABP')]
    if numpy.all(y[~numpy.isnan(y)] == 0): return
    i = numpy.arange(128, y.size, 128)
    return sample(y, i)

def extract(paths):
    path = paths[0].split('_')[0]
    y = {}
    for i in paths:
        try:
            y[i.split('/')[-1]] = extract_part(i)
        except:
            print(i)
    if len(y) > 0:
        with open(path + '.y128.pkl', 'wb') as f:
            pickle.dump(y, f)

if __name__ == '__main__':
    root = '/scr1/mimic/waveforms/'
    files = [
        i.split('.')[0] for i in os.listdir(root)
        if '.hea' in i and '_y' not in i and '_x' not in i
    ]
    paths = {i.split('_')[0] for i in files}
    paths = {k: [] for k in paths if not os.path.isfile(root + k + '.y128.pkl')}
    for i in files:
        k = i.split('_')[0]
        if k in paths:
            paths[k] += [root + i]

    Pool(cpu_count()).map(extract, paths.values())
