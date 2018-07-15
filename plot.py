import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')
from cycler import cycler

def plot_results(case,metric):
    """
    Fxn to plot results
    :param case: parameter to specify which data should be plotted. 'reg' for regression or 'clf' for classification
    :param metric: parameter for metric to be plotted. 'time', 'f1_score' or 'mae_score'
    :return: plot
    """
    check = pd.read_csv('./results_'+case+'.txt')
    check['version_sequence'] = check.groupby('library').cumcount()

    plt.rc('lines', linewidth=2)
    plt.rc('axes', prop_cycle=(cycler('color', ['orangered', 'mediumslateblue', 'teal', 'royalblue', 'gold'])))

    p = sns.tsplot(data=check, time="version_sequence", unit="library", condition="library", value=metric)
    txt=get_label_and_heading(case=case, metric=metric)[1]
    y_axis_label = get_label_and_heading(case=case, metric=metric)[0]
    p.set_ylabel(y_axis_label, fontsize=13)
    p.axes.set_title(txt, fontsize=14)
    p.set_xlabel("version", fontsize=13)
    p.set_xticks(np.arange(check['version_sequence'].nunique()))
    plt.style.use('bmh')
    p.legend(loc=0, fontsize=13)
    save_plot_to_file(plt, case, metric)
    plt.close()
    return None

def get_label_and_heading(case, metric):
    """
    helper fxn to get label & header
    :param case: (str) parameter to specify regression or classification case
    :param metric: (str) param to specify y-axis: time, f1_score, mae_score
    :return: (str) label and axis name
    """
    if metric == 'time':
        y_axis_label = 'Execution time (seconds)'
        if case == 'clf':
            txt = 'Classification execution time by version'
        if case == 'reg':
            txt = 'Regression execution time by version'
    if metric != 'time':
        if case == 'reg':
            txt = 'Mean absolute error score by version'
            y_axis_label = 'Mean absolute error'
        if case == 'clf':
            txt = 'F1 score by version'
            y_axis_label = 'F1 score'
    return y_axis_label, txt

def save_plot_to_file(p,case,metric):
    """
    helper fxn to save plots in dir
    :param p: matplot object
    :param case: (str) parameter to specify regression or classification case
    :param metric: (str) param to specify y-axis: time, f1_score, mae_score
    :return: None
    """
    if metric == 'time':
        if case == 'clf':
            p.savefig('./exec_time_clf.png')
        if case == 'reg':
            p.savefig('./exec_time_reg.png')
    if metric != 'time':
        if case == 'reg':
            p.savefig('./mae_reg.png')
        if case == 'clf':
            p.savefig('./f1_clf.png')
    return None


if __name__ == '__main__':
    plot_results(case='reg', metric='mae_score')
    plot_results(case='clf', metric='f1_score')
    plot_results(case='reg', metric='time')
    plot_results(case='clf', metric='time')


