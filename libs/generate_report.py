import numpy
import ipywidgets
from matplotlib import pyplot
import loss_metrics

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
    

def plot_diagnosis(fig, p_neg, p_pos, threshold, sensitivity, code):
    fig.clear()
    axis = fig.gca()
    axis.set_facecolor('black')
    bins = pyplot.hist(p_neg, bins=100, alpha=1, color='green');
    bins = pyplot.hist(p_pos, bins=100, alpha=.5, color='red');
    pyplot.plot([threshold] * 2, [0, 500], '--w')
    pyplot.xlabel('Probability', color='white')
    axis.tick_params(axis='x', colors='white')
    axis.tick_params(axis='y', colors='white')
    axis.set_xlim([0, 1])
    legend = ['Threshold', 'Negative', 'Positive']
    legend = fig.legend(legend, facecolor='black')
    for text in legend.get_texts():
        text.set_color('white')
    pct = round(sensitivity * 100, 2)
    title = get_code_name(code)
    subtitle = '{}% of cases detected'.format(pct)
    pyplot.title(title + '\n' + subtitle, color='white')    

def get_diagnoses_plotter(y_true, y_pred, codes):
    count = y_true.shape[1]
    
    sensitivities = [
        loss_metrics.precise_sensitivity(j, y_true, y_pred).numpy()
        for j in range(count)
    ]
    
    thresholds = [
        loss_metrics.precise_threshold(j, y_true, y_pred)
        for j in range(count)
    ]
    
    slider = ipywidgets.IntSlider(min=0, max=count-1, value=0);
    fig = pyplot.figure(1, facecolor='black');
    
    def update(i):
        i = numpy.argsort(sensitivities)[-i-1]
        plot_diagnosis(
            fig,
            y_pred[y_true[:, i] == -1, i], 
            y_pred[y_true[:, i] ==  1, i], 
            thresholds[i], 
            sensitivities[i], 
            codes[i]
        )
        pyplot.yscale('log')
        fig.canvas.draw()
        
    plot_diagnoses = lambda: ipywidgets.interact(update, i=slider)
    return plot_diagnoses
    