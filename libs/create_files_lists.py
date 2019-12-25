import pickle
import os
import flacdb

def get_downloaded_rec_segs():
    rec_segs = os.listdir(flacdb.ROOT)
    to_tuple = lambda i: flacdb.rec_seg_to_tuple(i.split('.')[0])
    S1 = {to_tuple(i) for i in rec_segs if '_x.flac' in i}
    S2 = {to_tuple(i) for i in rec_segs if '_x.hea' in i}
    S3 = {to_tuple(i) for i in rec_segs if '_y.hea' in i}
    S4 = {to_tuple(i) for i in rec_segs if '_y.flac' in i}
    S5 = {to_tuple(i) for i in rec_segs if '.hea' in i and '_x' not in i and '_y' not in i}
    S = S1 & S2 & S3 & S4 & S5
    return S

def get_rec_segs_with_bp():
    hdrs = pickle.load(open('/scr1/mimic/headers_.pkl', 'rb'))
    for i in hdrs:
        if i['sig_name'] is None:
            i['sig_name'] = []
    bp_sigs = ['ABP', 'CVP', 'PAP', 'ICP']
    hdrs_filtered = [
        i for i in hdrs
        if any(j in i['sig_name'] for j in bp_sigs)
        and 'n' not in i['record_name']    
    ]
    S = {flacdb.rec_seg_to_tuple(i['record_name']) for i in hdrs_filtered}
    return S

def get_matched_rec_segs():
    hdrs = pickle.load(open('/scr1/mimic/headers_matched_.pkl', 'rb'))
    S = {
        flacdb.rec_seg_to_tuple(i['record_name']) 
        for i in hdrs if i['record_name'][0] != 'p'
    }
    return S

if __name__ == '__main__':
    S_has_bp = get_rec_segs_with_bp()
    print(len(S_has_bp), 'record segments with bp')
    S_downloaded = get_downloaded_rec_segs()
    print(len(S_downloaded), 'record segments already downloaded')
#     S_matched = get_matched_rec_segs()
#     S = (S_has_bp | S_matched) - S_downloaded
    S = S_has_bp - S_downloaded
    print(len(S), 'record segments to download')
    rec_segs = sorted(S)
    _to_file = lambda x: '{}/{}_{}'.format(x[0], x[0], str(x[1]).zfill(4))
    _to_files = lambda x: [_to_file(x) + '.dat', _to_file(x) + '.hea']
    filenames = [j for i in rec_segs for j in _to_files(i)]
    for part in range(10):
        with open('/scr1/mimic/files_lists/part_{}.txt'.format(part), 'w') as f:
            f.write('\n'.join([i for i in filenames if i[1] == str(part)]))