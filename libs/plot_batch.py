from matplotlib import pyplot
from ipywidgets import interact, IntSlider


BP_SIGS = ['ABP', 'CVP', 'PAP', 'ICP', 'ART']

ECG_SIGS = ['I', 'II', 'III', 'V', 'AVR', 'AVF', 'AVL', 'MCL', 'MCL1']

SIG_COLORS = {
    **{k: 'red' for k in BP_SIGS},
    **{k: 'blue' for k in ECG_SIGS},
    'RESP': 'green',
    'PLETH': 'green',
}


def init(H, X, Y):    
    x, y = X['signals'].numpy(), Y['pressure'].numpy()
    fig, axes, lines = plot_example(H, x, y, 0)
    slider = IntSlider(min=0, max=(2 ** H['batch_size_log2'])-1, value=0)
    def update(i):
        return update_plot(H, fig, lines, axes, x, y, i)
    plotter = lambda: interact(update, i=slider)
    return plotter


def plot_example(H, x, y, i):
    fig = pyplot.figure(facecolor='black')
    fig.canvas.layout.height = '800px'
    fig.canvas.layout.width = '700px'
    pyplot.subplots_adjust(left=0.03, wspace=0, hspace=0.2)
    lines = {'sigs': {}, 'label': {}}
    axes = {}
    
    def plot_y(s, j, is_diastolic):
        xs = [0, 2 ** H['window_size_log2']]
        ys = [y[i, int(is_diastolic), j]] * 2
        line = axes[s].plot(xs, ys, '--', c='gray')[0]
        return line
    
    for j, s in enumerate(H['input_sigs']):
        axes[s] = pyplot.subplot(len(H['input_sigs']), 1, j + 1)
        line = axes[s].plot(x[i, :, j])[0]
        lines['sigs'][s] = line
        axes[s].set_ylabel(s)
        axes[s].set_facecolor('black')
        axes[s].yaxis.tick_right()
        if s in BP_SIGS:
            lines['label'][s] = {
                'sys': plot_y(s, j, is_diastolic=False), 
                'dia': plot_y(s, j, is_diastolic=True)
            }
    
    return fig, axes, lines

def update_plot(H, fig, lines, axes, x, y, i):
    for j, s in enumerate(H['input_sigs']):
        lines['sigs'][s].set_ydata(x[i, :, j])
        low, high = x[i, :, j].min(), x[i, :, j].max()
        dx = max(0.01, high - low)
        sig_color = SIG_COLORS[s] if x[i, :, j].any() else 'black'
        lines['sigs'][s].set_color(sig_color)
        axes[s].set_ylim(bottom = low - 0.1 * dx, top = high + 0.1 * dx)
        if s in H['output_sigs']:
            line = lines['label'][s]
            line['sys'].set_ydata([y[i, 0, j]] * 2)
            line['dia'].set_ydata([y[i, 1, j]] * 2)
            line['sys'].set_color('gray' if y[i, :, j].any() else 'black')
            line['dia'].set_color('gray' if y[i, :, j].any() else 'black')
        
        axis = fig.axes[j]
        axis.yaxis.label.set_color('white' if x[i, :, j].any() else 'dimgray')
        axis.tick_params(axis='y', colors='gray')
        if x[i, :, j].any():
            ticks = [x[i, :, j].min(), x[i, :, j].max()]
        else:
            ticks = []
        axis.set_yticks(ticks)
        
    fig.canvas.draw()
    pyplot.show()