import os
import pickle
import numpy
from scipy import signal

DB_ROOT = '/scr-ssd/mimic/'

def load_validation_estimates(path):
    with open(path, 'rb') as f:
        estimates = pickle.load(f)
    return estimates

def load_validation_template():
    with open(DB_ROOT + 'validation-template.pkl', 'rb') as f:
        template = pickle.load(f)
    return template

def save_validation_estimates(estimates, path):
    template = load_validation_template()
    assert(template.keys() == estimates.keys())
    for record in template:
        assert(estimates[record].shape == template[record].shape)
    for record in estimates:
        if estimates[record].dtype != 'float16':
            estimates[record] = estimates[record].astype('float16')
        assert(~numpy.isnan(estimates[record]).any())
    with open(path, 'wb') as f:
        pickle.dump(estimates, f)

def compute_patient_loss(targets, estimates):
    targets = numpy.concatenate(targets)
    estimates = numpy.concatenate(estimates)
    residuals = numpy.subtract(targets, estimates, dtype='float64')
    losses = numpy.abs(residuals[~numpy.isnan(residuals)])
    loss = losses.mean()
    return loss

def compute_loss(targets, estimates):

    records = {}
    for record in estimates:
        patient = record.split('_')[0]
        if patient in records:
            records[patient].append(record)
        else:
            records[patient] = [record]

    losses = []
    for patient in records:
        y = [targets[record] for record in records[patient]]
        y_ = [estimates[record] for record in records[patient]]
        losses.append(compute_patient_loss(y, y_))

    loss = numpy.array(losses).mean()
    return loss

def compute_validation_loss(estimates):
    template = load_validation_template()
    patients = sorted({i.split('_')[0] for i in template.keys()})
    targets = load_targets(patients, 'validation', reject_nan=True)
    targets = {i: targets[i] for i in targets if i in template}
    loss = compute_loss(targets, estimates)
    return loss

def filter_target_noise(y, w):
    w = min(w, y.shape[0])
    w -= w % 2 == 0
    return signal.savgol_filter(y, w, 1, axis=0)

def load_targets(patients, part, filter_window=300, min_length=5, reject_nan=False):
    targets = {}
    for patient in patients:
        path = DB_ROOT + part + '/' + str(patient) + '.y.pkl'
        with open(path, 'rb') as f:
            targets.update(pickle.load(f))
    targets = {
        k: filter_target_noise(v, filter_window) if filter_window else v
        for k, v in targets.items()
        if v is not None and v.shape[0] >= min_length
    }
    targets = {
        k: v for k, v in targets.items()
        if not (numpy.isnan(v).any() if reject_nan else numpy.isnan(v).all())
    }
    return targets

def initialize_validation_template(input_sigs):
    files = os.listdir(DB_ROOT + 'validation')
    patients = sorted({i.split('.')[0] for i in files if '.y.pkl' in i})
    rec_names = {
        i.split('.')[0] for i in files
        if '_' in i and i[i.index('.') - 1] not in ['x', 'y']
    }
    targets = load_targets(patients, 'validation', reject_nan=True)

    with open(DB_ROOT + 'headers.pkl', 'rb') as f: hdrs = pickle.load(f)

    keep_recs = {
        i['record_name'] for i in hdrs
        if i['record_name'] in rec_names
        and all(j in i['sig_name'] for j in input_sigs)
    }

    template = {
        i: ~numpy.isnan(targets[i])
        for i in targets if i in keep_recs
    }

    with open(DB_ROOT + 'validation-template.pkl', 'wb') as f:
        pickle.dump(template, f)
