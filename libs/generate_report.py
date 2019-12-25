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

import tensorflow as tf

def compute_predictions(model, dataset, batch_count, fix_input=None):
    X, Y, P = [], [], []
    for x, y in dataset.take(batch_count):
        X.append(x)
        Y.append(y)
        if fix_input is not None:
            x = fix_input(x)
        P.append(model.predict(x))
    keys = enumerate(['pressure', 'diagnosis'])
    P = {k: numpy.concatenate([p[i] for p in P]) for i, k in keys}
    X = {k: numpy.concatenate([x[k] for x in X]) for k in X[0].keys()}
    Y = {k: numpy.concatenate([y[k] for y in Y]) for k in Y[0].keys()}
    return X, Y, P


def get_code_name(code):
    if code in loss_metrics.CODE_NAMES:
        name = loss_metrics.CODE_NAMES[code]
    else:
        name = code
    name = name.replace('_', ' ').title()
    return name    
    

def plot_diagnosis(axis, p_neg, p_pos, threshold, sensitivity, name):
    axis.clear()
    bins = numpy.linspace(0, 1, 101)
    axis.hist(p_neg, bins=bins, alpha=1, color='green');
    axis.hist(p_pos, bins=bins, alpha=.5, color='red');
    axis.plot([threshold] * 2, [0, 2**10], '--', color='white')
#     axis.set_xlabel('Probability', color='white')
    axis.tick_params(axis='x', colors='white')
    axis.tick_params(axis='y', colors='white')
    axis.set_xlim([0, 1])
    legend = ['Threshold', 'Negative', 'Positive']
    legend = axis.legend(legend, facecolor='black')
    for text in legend.get_texts():
        text.set_color('white')
    pct = round(sensitivity * 100, 2)
    subtitle = '{}% of cases detected'.format(pct)
    axis.set_title(name + '\n' + subtitle, color='white')    

    
def get_diagnoses_plotter(x, y_true, y_pred, codes, subject_ids):    
    precisions = [0.75, 0.8, 0.9, 0.95, 0.99]
    
    condition_names = [get_code_name(code) for code in codes]
    
    sensitivities = {p: {
        name: loss_metrics.precise_sensitivity(j, y_true, y_pred, p).numpy()
        for j, name in enumerate(condition_names)
    } for p in precisions}
    
    thresholds = {p: {
        name: loss_metrics.precise_threshold(j, y_true, y_pred, p)
        for j, name in enumerate(condition_names)
    } for p in precisions}
    
    scores = [sensitivities[0.9][name] for name in condition_names]
    I = numpy.argsort(scores)[::-1]
    
    condition_slider = ipywidgets.SelectionSlider(
        options = [condition_names[i] for i in I],
        description = 'Condition:',
        continuous_update = False,
        readout = True,
        layout = ipywidgets.Layout(width='90%')
    )
    
    example_slider = ipywidgets.IntSlider(
        min = 0,
        max = x.shape[0] - 1,
        continuous_update = False,
        description = 'Example:',
        layout = ipywidgets.Layout(width='90%')
    )
    
    precision_slider = ipywidgets.SelectionSlider(
        options = precisions,
        value = 0.9,
        description = 'Precision:',
        continuous_update = False,
        readout = True
    )

    
    logscale_button = ipywidgets.ToggleButton(
        value=False,
        description='Log Scale',
    )
    
    fig, axes = pyplot.subplots(
        nrows=4, 
        facecolor='black', 
        gridspec_kw={'height_ratios': [8, 1, 2, 2]}
    )
    
    for axis in axes:
        axis.set_facecolor('black')
    
    spines = list(axes[0].spines.values())
    spines[0].set_edgecolor('white')
    spines[2].set_edgecolor('white')
    
    axes[1].plot(list(range(10)), 'k')
    
    for axis in axes[2:]:
        axis.yaxis.tick_right()
        axis.tick_params(axis='y', colors='gray')
    
    def update(name, precision, example_index, log_scale):
        j = condition_names.index(name)
        
        plot_diagnosis(
            axes[0],
            y_pred[y_true[:, j] == -1, j], 
            y_pred[y_true[:, j] ==  1, j], 
            thresholds[precision][name], 
            sensitivities[precision][name], 
            name
        )
        
        if log_scale:
            axes[0].set_yscale('log')
        else:
            axes[0].set_yscale('linear')
                
        i = numpy.argsort(-y_pred[:, j])[example_index]
        for s in [0, 1]:
            axes[s+2].clear()
            axes[s+2].plot(x[i, :, s], ['b', 'y'][s])
        
        colors = {-1: 'g', 0: 'b', 1: 'r'}
        height = 2**5 if log_scale else 2**9
        bar_x, bar_y = [y_pred[i, j]] * 2, [0, height]
        axes[0].plot(bar_x, bar_y, '-', color='white')
        axes[0].plot(bar_x, bar_y, '--' + colors[y_true[i, j]])
        
        axes[2].set_ylabel('PLETH', color='white')
        axes[3].set_ylabel('II', color='white')
        axes[3].set_xlabel('Subject ' + str(subject_ids[i]), color='white')
        
        pyplot.tight_layout(pad=1)
        pyplot.subplots_adjust(hspace=0.3)
        fig.canvas.layout.height = '600px'
        fig.canvas.draw()
        
    plot_diagnoses = lambda: ipywidgets.interact(
        update,
        name = condition_slider,
        precision= precision_slider,
        example_index = example_slider,
        log_scale = logscale_button
    )
    
    return fig, plot_diagnoses


def get_predictions(H, model, dataset, weights_path, batch_count):
    
    fix_input = lambda x: {**x, 'mask': tf.cast(x['mask'], 'float')}
    if os.path.isfile(weights_path):
        print('loading predictions')
        with open(checkpoint_path + '.pkl', 'rb') as f:
            X, Y, P = pickle.load(f)
    else:
        print('computing predictions')
        X, Y, P = compute_predictions(model, dataset, batch_count, fix_input)
        with open(weights_path + '.pkl', 'wb') as f:
            pickle.dump([X, Y, P], f)
    return X, Y, P


def run(model_id, checkpoint_index, example_count_log2):
    
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
    tensors, metadata, priors = initialize.run(H, parts=[part])
    dataset = data_pipeline.build(H, tensors[part], part)
    
    batch_count = 2 ** (example_count_log2 - H['batch_size_validation_log2'])
    model = conv_model.build(H, priors)
    model.load_weights(weights_path)
    X, Y, P = get_predictions(H, model, dataset, weights_path, batch_count)
            
    y_true, y_pred = Y['diagnosis'], P['diagnosis']
    x = X['signals'][:, :, [H['input_sigs'].index(i) for i in ['PLETH', 'II']]]
    M = metadata.reset_index()[['subject_id', 'rec_id']].drop_duplicates()
    M = M.set_index('rec_id', verify_integrity=True)
    subject_ids = M.loc[Y['rec_id']].values[:, 0]
    codes = priors.index.to_list()
    fig, plotter = get_diagnoses_plotter(x, y_true, y_pred, codes, subject_ids)
    
    return plotter, model