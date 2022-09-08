
import numpy as np
import matplotlib.pyplot as plt
import datetime


def plot_stuff(results, methods, dataset):

    font = {'size': 43}
    plt.rc('font', **font)
    my_dpi = 100
    plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)

    for idx, curr_method in enumerate(methods):

            if curr_method == 'ITL':
                color = 'black'
                linestyle = '--'
                linewidth = 3
                curr_method_short = 'ITL'
            elif curr_method == 'unconditional_Bias':
                color = 'tab:green'
                linestyle = '-'
                linewidth = 3
                curr_method_short = 'uncond. BIAS'
            elif curr_method == 'conditional_Bias':
                color = 'tab:purple'
                linestyle = '-'
                linewidth = 3
                curr_method_short = 'cond. BIAS'
            elif curr_method == 'unconditional_Feature':
                color = 'tab:blue'
                linestyle = '-'
                linewidth = 3
                curr_method_short = 'uncond. FEAT'
            elif curr_method == 'conditional_Feature':
                color = 'tab:pink'
                linestyle = '-'
                linewidth = 3
                curr_method_short = 'cond. FEAT'
            elif curr_method == 'unconditional_Mahalanobis':
                color = 'tab:red'
                linestyle = '-'
                linewidth = 3
                curr_method_short = 'uncond. MAHA'
            elif curr_method == 'conditional_Mahalanobis':
                color = 'c'
                linestyle = '-'
                linewidth = 3
                curr_method_short = 'cond. MAHA'

            mean = np.nanmean(results[curr_method], axis=0)
            std = np.nanstd(results[curr_method], axis=0)

            plt.plot(mean, color=color, linestyle=linestyle, linewidth=linewidth, label=curr_method_short)
            plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color, antialiased=True)
            plt.xlabel('Training Tasks', fontsize=50, fontweight="normal")
            plt.ylabel('Test Error', fontsize=50, fontweight="normal")
            plt.legend(loc='upper right')
            # plt.legend(loc='upper right')

            # plt.ylim(bottom=0, top=8)
            plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(dataset + '_' + 'temp_test_error' + '_' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    plt.pause(0.01)
    plt.close()


