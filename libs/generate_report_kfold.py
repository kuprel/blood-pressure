import numpy
import json
import pickle
import os
import ipywidgets
from matplotlib import pyplot
import loss_metrics
import initialize
import data_pipeline
import conv_model
import icd_util
from sklearn.metrics import roc_auc_score
import tensorflow as tf

def compute_predictions(model, dataset, batch_count, fix_input=None):
    X, Y, P = [], [], []
    for x, y in dataset.take(batch_count):
        X.append(x)
        Y.append(y)
        if fix_input is not None:
            x = fix_input(x)
        P.append(model.predict(x))
    X = {k: numpy.concatenate([x[k] for x in X]) for k in X[0].keys()}
    Y = {k: numpy.concatenate([y[k] for y in Y]) for k in Y[0].keys()}
    P = numpy.concatenate(P)
    return X, Y, P
    

def plot_diagnosis(axis, y_true, y_pred, threshold, name, dark):
    p_neg = y_pred[y_true == -1] 
    p_pos = y_pred[y_true ==  1]
    p_all = y_pred[y_true !=  0]
    axis.clear()
    bins = numpy.linspace(0, 1, 101)
    axis.hist(p_neg, bins=bins, alpha=1 if dark else 0.5, color='green');
    axis.hist(p_pos, bins=bins, alpha=0.5, color='red');
    axis.plot([threshold] * 2, [0, 2**8], '--', color='white' if dark else 'black')
    axis.set_xlim([0, 1])
    prior = len(p_pos) / len(p_all)
    prior = round(prior * 100, 1)
    precision = sum(p_pos > threshold) / max(1, sum(p_all > threshold))
    precision = round(precision * 100, 1)
    sensitivity = sum(p_pos > threshold) / len(p_pos)
    sensitivity = round(sensitivity * 100, 1)
    auc = loss_metrics._roc_auc(y_true, y_pred).numpy()
    auc = round(auc * 100, 1)
    sub1 = '{}% prior -> {}% precison'.format(prior, precision)
    sub2 = '{}% detected'.format(sensitivity)
    sub3 = '{}% AUC'.format(auc)
    axis.set_title(name + '\n' + sub1 + ', ' + sub2 + '\n' + sub3)
#     axis.set_xlabel('Probability')

    
def get_plotter(x, y_true, y_pred, codes, subject_ids, sigs, dark):    
        
    group_names = icd_util.load_group_strings()
    
    def get_code_name(code):
        if code not in group_names:
            return code
        name = group_names[code].replace("'", '').title()
        return name
        
    scores = [
        loss_metrics.roc_auc_score(j, y_true, y_pred).numpy()
        for j in range(y_true.shape[1])
    ]
    I = numpy.argsort(scores)[::-1]
    
    condition_slider = ipywidgets.SelectionSlider(
        options = [codes[i] for i in I],
        description = 'Condition:',
        continuous_update = False,
        readout = True,
        layout = ipywidgets.Layout(width='90%')
    )
    
    example_slider = ipywidgets.IntSlider(
        min = 0,
        max = x.shape[0] - 1,
        value = 3 * x.shape[0] // 4,
        continuous_update = False,
        description = 'Example:',
        layout = ipywidgets.Layout(width='90%')
    )
    
    threshold_slider = ipywidgets.SelectionSlider(
        options = numpy.arange(0, 1, 0.005),
        value = 0.5,
        description = 'Threshold:',
        continuous_update = False,
        readout = True,
        layout = ipywidgets.Layout(width='90%')
    )
    
    logscale_button = ipywidgets.ToggleButton(
        value=False,
        description='Log Scale',
    )
    
    if dark:
        pyplot.style.use('dark_background')
    
    fig, axes = pyplot.subplots(
        nrows=len(sigs) + 2,
        gridspec_kw={'height_ratios': [8, 2] + [2] * len(sigs)}
    )

    for axis in axes[1:]:
        for spine in axis.spines.values():
            spine.set_visible(False)
    
    axes[1].plot(list(range(10)), 'k' if dark else 'w')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    for axis in axes[2:]:
        axis.yaxis.tick_right()
        axis.tick_params(axis='y', colors='gray')
        axis.set_xticks([])
        
    I = [numpy.argsort(y_pred[:, j]) for j in range(len(codes))]
    
    def update(code, threshold, example_index, log_scale):
        j = codes.index(code)
                
        i = I[j][example_index]
        for s in range(len(sigs)):
            axes[s+2].clear()
            axes[s+2].plot(x[i, :, s], ['b', 'y', 'y', 'y', 'g', 'r'][s])
            axes[s+2].set_ylabel(sigs[s])
            axes[s+2].set_xticks([])

        axes[1].set_xlabel('Subject ' + str(subject_ids[i]))
        
        plot_diagnosis(
            axes[0],
            y_true[:, j],
            y_pred[:, j],
            threshold,
            get_code_name(code),
            dark
        )
        
        axes[0].set_yscale('log' if log_scale else 'linear')
    
        colors = {-1: 'gray', 0: 'gray', 1: 'white' if dark else 'black'}
        color = colors[y_true[i, j]]
        axes[0].plot(y_pred[i, j], 1, color=color, marker=7, markersize=4)
        
        pyplot.tight_layout(pad=1)
        pyplot.subplots_adjust(hspace=0.3)
        fig.canvas.layout.height = '1000px'
        fig.canvas.draw()
        
    plot_diagnoses = lambda: ipywidgets.interact(
        update,
        code = condition_slider,
        threshold = threshold_slider,
        example_index = example_slider,
        log_scale = logscale_button
    )
    
    return fig, plot_diagnoses


def get_predictions(H, priors, dataset, weights_path, batch_count):
    
    fix_input = lambda x: {**x, 'mask': tf.cast(x['mask'], 'float')}
    if os.path.isfile(weights_path + '.pkl'):
        print('loading predictions')
        with open(weights_path + '.pkl', 'rb') as f:
            X, Y, P = pickle.load(f)
    else:
        print('loading model')
        model = conv_model.build(H, priors)
        model.load_weights(weights_path)
        print('computing predictions')
        X, Y, P = compute_predictions(model, dataset, batch_count, fix_input)
        with open(weights_path + '.pkl', 'wb') as f:
            pickle.dump([X, Y, P], f)
    return X, Y, P


def generate_predictions(model_id, fold_index, checkpoint_index, example_count_log2):
    
    ckpts = os.listdir('/scr1/checkpoints')
    ckpts = sorted(i for i in ckpts if 'index' in i and str(model_id) in i)
    hypes_path = '../hypes/{}.json'.format(ckpts[0].split('.')[0][:-6])
    weights_path = '/scr1/checkpoints/' + ckpts[checkpoint_index]
    assert(os.path.isfile(hypes_path) and os.path.isfile(weights_path))
    weights_path = weights_path.replace('.index', '')
    print('found hypes', hypes_path, '\nfound weights', weights_path)
    
    H0 = json.load(open(hypes_path))
    H = initialize.load_hypes()
    H = {**H, **H0}
    H['batch_size_validation_log2'] = 7

    part = 'validation'
    path = '/scr1/mimic/initial_data_{}_fold_{}/'.format(model_id, fold_index)
    tensors, metadata, priors = initialize.run(H, parts=[part], load_path=path)
    dataset = data_pipeline.build(H, tensors[part], part)
    
    batch_count = 2 ** (example_count_log2 - H['batch_size_validation_log2'])
    
    X, Y, P = get_predictions(H, priors, dataset, weights_path, batch_count)
            
    return H, X, Y, P, metadata, priors
    
def generate_plotter(H, X, Y, P, metadata, priors, dark=True):
    y_true, y_pred = Y['diagnosis'], P
    sigs = ['PLETH', 'II', 'V', 'AVR', 'RESP', 'ABP']
    sig_index = [H['input_sigs'].index(i) for i in sigs]
    x = X['signals'][:, :, sig_index]
    M = metadata.reset_index()[['subject_id', 'rec_id']].drop_duplicates()
    M = M.set_index('rec_id', verify_integrity=True)
    subject_ids = M.loc[Y['rec_id']].values[:, 0]
    codes = priors.index.to_list()
    print(y_pred.shape, 'predictions shape')
    
    fig, plotter = get_plotter(x, y_true, y_pred, codes, subject_ids, sigs, dark)
    return plotter
    
    