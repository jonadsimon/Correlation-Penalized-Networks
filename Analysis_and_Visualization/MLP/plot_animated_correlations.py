# Used code from two places:
# 1) http://stackoverflow.com/questions/25161449/make-a-matplotlib-animation-by-clfing-each-frame
# 2) http://stackoverflow.com/questions/8822370/plot-line-graph-from-histogram-data-in-matplotlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import timeit

start_time = timeit.default_timer()

n_hidden_units = 124750 # = (500^2 - 500)/2
n_epochs = 100
framerate = 5
bin_width = 0.05
hist_range = [0.0,1.0] # [-0.10,1.10]
bin_edges = np.arange(hist_range[0], hist_range[1], bin_width)
bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

input_dir = os.path.join(os.path.split(__file__)[0], '..', '..', 'output', 'MLP')

###########################################################
### Read in the files containing the validations losses ###
###########################################################

validLoss0_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000000_Drop1.000000.csv'
validLoss1e5_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000010_Drop1.000000.csv'
validLoss1e4_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.000100_Drop1.000000.csv'
validLoss1e3_file = 'ValidationLoss_Epoch100_Batch250000_Cor0.001000_Drop1.000000.csv'

validLoss0_df = pd.read_csv(input_dir+'/'+validLoss0_file)
validLoss1e5_df = pd.read_csv(input_dir+'/'+validLoss1e5_file)
validLoss1e4_df = pd.read_csv(input_dir+'/'+validLoss1e4_file)
validLoss1e3_df = pd.read_csv(input_dir+'/'+validLoss1e3_file)


###############################################################
### Read in the files containing the flattened correlations ###
###############################################################

flatCorr0_file = 'FlatCorrelations_Epoch100_Batch250000_Cor0.000000_Drop1.000000.csv'
flatCorr1e5_file = 'FlatCorrelations_Epoch100_Batch250000_Cor0.000010_Drop1.000000.csv'
flatCorr1e4_file = 'FlatCorrelations_Epoch100_Batch250000_Cor0.000100_Drop1.000000.csv'
flatCorr1e3_file = 'FlatCorrelations_Epoch100_Batch250000_Cor0.001000_Drop1.000000.csv'

def get_bar_heights(flatcorr_filepath):
    bar_heights = []
    with open(flatcorr_filepath) as infile:
        for line in infile.readlines():
            abs_corrs = map(lambda x: abs(float(x)), line.split(',')[1:])
            heights, _ = np.histogram(abs_corrs, bins=bin_edges)
            scaled_heights = 1.0*heights / n_hidden_units # scale by total number of hidden unit pairs
            bar_heights.append(scaled_heights)
            
    return bar_heights

cor0_bar_heights = get_bar_heights(input_dir+'/'+flatCorr0_file)
cor1e5_bar_heights = get_bar_heights(input_dir+'/'+flatCorr1e5_file)
cor1e4_bar_heights = get_bar_heights(input_dir+'/'+flatCorr1e4_file)
cor1e3_bar_heights = get_bar_heights(input_dir+'/'+flatCorr1e3_file)

fig = plt.figure(figsize=(15, 12))

cor0_ax = plt.subplot2grid((3,2), (0,0))
cor1e5_ax = plt.subplot2grid((3,2), (0,1))
cor1e4_ax = plt.subplot2grid((3,2), (1, 0))
cor1e3_ax = plt.subplot2grid((3,2), (1, 1))
validloss_ax = plt.subplot2grid((3,2), (2, 0), colspan=2)


def gen_new_corr_plot(axis_handle, plot_title, bar_heights, this_iter):
    axis_handle.clear()
    axis_handle.set_xlim(hist_range[0], hist_range[1]) # possible to pass this in as a parameter?
    axis_handle.set_ylim(0, 1)
    axis_handle.set_xlabel("Absolute Pairwise Correlation")
    axis_handle.set_ylabel("Fraction of Unit Pairs")
    axis_handle.set_title(plot_title)
    p = axis_handle.bar(bin_centers, bar_heights[this_iter], align='center', width=bin_width)


def gen_new_error_plot(axis_handle, label_list, df_list, this_iter):
    axis_handle.clear()
    axis_handle.set_xlim(0, n_epochs)
    axis_handle.set_ylim(0, 0.10)
    axis_handle.set_xlabel("Epoch")
    axis_handle.set_ylabel("Validation Error")
    axis_handle.set_title("Validation Error During Training")
    for i in range(len(df_list)):
        axis_handle.plot(df_list[i]["Epoch"][:(this_iter+1)], df_list[i]["Error"][:(this_iter+1)], linewidth=1.5, label=label_list[i])
    axis_handle.legend()


def updatefig(i):
    gen_new_corr_plot(cor0_ax, 'Correlation Penalty = 0', cor0_bar_heights, i)
    gen_new_corr_plot(cor1e5_ax, 'Correlation Penalty = 1e-5', cor1e5_bar_heights, i)
    gen_new_corr_plot(cor1e4_ax, 'Correlation Penalty = 1e-4', cor1e4_bar_heights, i)
    gen_new_corr_plot(cor1e3_ax, 'Correlation Penalty = 1e-3', cor1e3_bar_heights, i)

    this_label_list = ["cor_reg=0.0", "cor_reg=1e-5", "cor_reg=1e-4", "cor_reg=1e-3"]
    this_df_list = [validLoss0_df, validLoss1e5_df, validLoss1e4_df, validLoss1e3_df]
    gen_new_error_plot(validloss_ax, this_label_list, this_df_list, i)
    
    plt.tight_layout()
    plt.draw()


outfile_name = 'FlatCorrelationsAndValidationLoss_BatchSize20_Epoch%i.mp4' % n_epochs
outfile_path = os.path.join(os.path.split(__file__)[0], outfile_name)
anim = animation.FuncAnimation(fig, updatefig, n_epochs)
anim.save(outfile_path, fps=framerate, bitrate=-1, codec="libx264")

end_time = timeit.default_timer()
print "Animation generation with %i epochs took %.2fm" % (n_epochs, (end_time - start_time)/60.0)
