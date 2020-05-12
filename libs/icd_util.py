import json
import pandas
import load_diagnosis


def _to_string(i, with_dot):
    a = str(i[0]).zfill(3)
    b = ''.join(str(j) for j in i[1:])
    dot = '.' if with_dot and len(i) > 1 else ''
    return a + dot + b

def code_tuple_to_string(i):
    return _to_string(i, with_dot=False)

def code_string_to_tuple(i):
    return (int(i[:3]), *[int(j) for j in i[3:]])

def group_string_to_tuple(i):
    if '-' in i:
        return tuple((int(j),) for j in i.split('-'))
    elif '.' in i:
        a, b = i.split('.')
        return ((int(a), *[int(j) for j in b]),)
    else:
        return ((int(i),),)

def group_tuple_to_string(i):
    return '-'.join(_to_string(j, with_dot=True) for j in i)

def is_contained(code, group):
    if len(group) == 1:
        node = group[0]
        if node[0] != code[0] or len(node) > len(code):
            return False
        return all(node[i] == code[i] for i in range(len(node)))
    else:
        return group[0][0] <= code[0] <= group[1][0]

def load_data():
    return json.load(open('/scr-ssd/icd/icd9.json'))

def load_group_strings():
    icd_data = load_data()
    group_strings = {
        j['code']: j['descr'].title()
        for i in icd_data for j in i 
        if j['code'] is not None and j['code'][0] in '0123456789'   
    }
    return group_strings

def get_group_counts():
    code_strings = load_diagnosis.load_code_strings()
    code_tuples = {code_string_to_tuple(i) for i in code_strings}
    
    groups = load_group_strings()
    groups = {group_string_to_tuple(i): groups[i] for i in groups}
    
    code_groups = {
        c: [g for g in groups if is_contained(c, g)] 
        for c in code_tuples
    }
    
    code_diagnosis = load_diagnosis.load_raw()
    group_diagnosis = {}
    for i in code_diagnosis.index.get_level_values('subject_id').unique():
        for j in code_diagnosis.loc[i].index.get_level_values('hadm_id').unique():
            group_diagnosis[i, j] = set()
            for k in code_diagnosis.loc[(i, j)].index:
                if type(k) is str and k[0] in '0123456789':
                    code = code_string_to_tuple(k)
                    group_diagnosis[i, j].update(code_groups[code])
    
    group_counts = {group: 0 for group in groups}
    for i in group_diagnosis:
        for group in group_diagnosis[i]:
            group_counts[group] += 1
    
    df = pandas.DataFrame.from_dict({
        'name': {group_tuple_to_string(k): groups[k] for k in groups},
        'count': {group_tuple_to_string(k): group_counts[k] for k in group_counts}
    })
    df.sort_values(['count'], ascending=False, inplace=True)
    return df

def test():
    code_strings = load_diagnosis.load_code_strings()
    code_tuples = {code_string_to_tuple(i) for i in code_strings}

    group_strings = load_group_strings()
    group_tuples = {group_string_to_tuple(i) for i in group_strings}

    print(len(code_tuples), 'codes')
    assert(len(code_tuples) == len(code_strings))

    assert(all(
        code_tuple_to_string(code_string_to_tuple(i)) == i 
        for i in code_strings
    ))

    print(len(group_tuples), 'groups')
    assert(len(group_tuples) == len(group_strings))

    assert(all(
        group_tuple_to_string(group_string_to_tuple(i)) == i 
        for i in group_strings
    ))