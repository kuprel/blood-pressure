import pandas
import numpy
import icd_util

clinic_file = lambda i: '/scr1/mimic/clinic/{}.csv'.format(i.upper())


def load_raw():
    diagnosis = pandas.read_csv(clinic_file('diagnoses_icd'))
    new_names = {i.upper(): i for i in ['subject_id', 'hadm_id']}
    new_names['ICD9_CODE'] = 'code'
    diagnosis = diagnosis[new_names].rename(columns=new_names)
    diagnosis.drop_duplicates(inplace=True)
    diagnosis.set_index(['subject_id', 'hadm_id', 'code'], inplace=True)
    diagnosis.sort_index(inplace=True)
    return diagnosis

def load_raw_admission():
    diagnosis = pandas.read_csv(clinic_file('admissions'))
    new_names = {i.upper(): i for i in ['subject_id', 'hadm_id']}
    new_names['DIAGNOSIS'] = 'admission_diagnosis'
    diagnosis = diagnosis[new_names].rename(columns=new_names)
    fixed = diagnosis['admission_diagnosis'].map(lambda i: str(i).title())
    diagnosis['admission_diagnosis'] = fixed
    diagnosis.drop_duplicates(inplace=True)
    diagnosis.set_index(
        ['subject_id', 'hadm_id', 'admission_diagnosis'], 
        inplace=True
    )
    diagnosis.sort_index(inplace=True)
    return diagnosis

def load_code_strings():
    diagnosis = load_raw()
    codes = diagnosis.reset_index()['code'].unique()
    codes = {i for i in codes if type(i) is str and i[0] in '0123456789'}
    return codes

def load_grouped():
    code_strings = load_code_strings()
    code_tuples = {icd_util.code_string_to_tuple(i) for i in code_strings}
    
#     groups = open('selected_icd_groups.txt').readlines()
#     groups = [i.split()[0] for i in groups]
    path = '/sailhome/kuprel/blood-pressure/diagnosis_codes.csv'
    groups = pandas.read_csv(path, sep='\t')
    groups = groups[groups['use'] == '+']
    groups = [i.strip() for i in groups['index']]
    groups = [icd_util.group_string_to_tuple(i) for i in groups]
    
    code_groups = {
        c: [g for g in groups if icd_util.is_contained(c, g)] 
        for c in code_tuples
    }
    
    code_diagnosis = load_raw()
    group_diagnosis = {}
    for i in code_diagnosis.index.get_level_values('subject_id').unique():
        for j in code_diagnosis.loc[i].index.get_level_values('hadm_id').unique():
            group_diagnosis[i, j] = set()
            for k in code_diagnosis.loc[(i, j)].index:
                if type(k) is str and k[0] in '0123456789':
                    code = icd_util.code_string_to_tuple(k)
                    group_diagnosis[i, j].update(code_groups[code])

    
    selected_group_diagnosis = {
        i: {
            icd_util.group_tuple_to_string(j) 
            for j in group_diagnosis[i] if j in groups
        }
        for i in group_diagnosis
    }

    data = [
        [i[0], i[1], j]
        for i in selected_group_diagnosis
        for j in selected_group_diagnosis[i]
    ]
    
    diagnosis = pandas.DataFrame(data, columns=['subject_id', 'hadm_id', 'code'])
    diagnosis.set_index(['subject_id', 'hadm_id', 'code'], inplace=True)
    diagnosis.sort_index(inplace=True)
    diagnosis.at[:, 'present'] = True
    diagnosis = diagnosis.unstack(fill_value=False)['present'].astype('bool')
    return diagnosis

def augment(diagnosis, metadata):
    matched_data = metadata.reset_index()
#     matched_data = matched_data[matched_data['subject_id'] > 0]
#     matched_data = matched_data[matched_data['hadm_id'] > 0]
    matched_data.drop_duplicates(diagnosis.index.names, inplace=True)
    matched_data.set_index(diagnosis.index.names, inplace=True)
    matched_data.sort_index(inplace=True)
        
    diagnosis = diagnosis.reindex(matched_data.index)
    
    diagnosis = diagnosis[diagnosis.notna().all(1)]
    matched_data = matched_data.reindex(diagnosis.index)
    
#     for i in ['gender', 'race']:
    for i in ['gender']:
        values = matched_data[i]
        for j in values.dtype.categories:
#         for j in ['F']:
            diagnosis.at[values.notna(), i + '_' + j] = values == j
    
#     thresholds = {'age': 75, 'height': 70, 'weight': 100}
    thresholds = {'age': 75}
    
    for k in thresholds:
        is_int = numpy.issubdtype(matched_data[k].dtype, numpy.signedinteger)
        is_given = matched_data[k] > 0 if is_int else matched_data[k].notna()
        name = k + '_at_least_' + str(round(thresholds[k]))
        diagnosis.at[is_given, name] = matched_data[k] >= thresholds[k]
    
    died = matched_data['death_time'].notna()
    diagnosis.at[(slice(None), slice(0, None)), 'died'] = died
    
#     bool_to_int = {True: 1, False: -1, numpy.nan: 0}
    bool_to_int = {True: 1, False: 0, numpy.nan: 0}
    diagnosis = diagnosis.replace(bool_to_int).astype('int8')
    
    return diagnosis


def fix(diagnosis):
    is_negative_always = (diagnosis == -1).all(level=0)
    is_diagnosed_always = (diagnosis.index.to_frame()['hadm_id'] > 0).all(level=0)
    is_negative = is_negative_always[is_diagnosed_always]
    diagnosis = diagnosis.replace({-1: 0})
    I = is_negative.index
    diagnosis.loc[I] = diagnosis.loc[I].mask(is_negative, -1)
#     diagnosis = diagnosis.drop(columns='other')
    return diagnosis.astype('int8')


def conform(diagnosis, metadata):
    unindexed = metadata.reset_index()
    I = pandas.MultiIndex.from_frame(unindexed[diagnosis.index.names])
    diagnosis = diagnosis.reindex(I).fillna(-1).astype('int8').reset_index()
    frames = [diagnosis, unindexed[['rec_id', 'seg_id']]]
    diagnosis = pandas.concat(frames, sort=False, axis=1)
    diagnosis = diagnosis.set_index(metadata.index.names)
    diagnosis = diagnosis.drop(columns='hadm_id')
    assert((diagnosis.index == metadata.index).all())
    return diagnosis
