import matplotlib.pylab as plt
import math
import numpy as np
import random

'''
colour functions are from https://gist.github.com/adewes/5884820
'''
def pandas_collumn_to_numpy_array(pandas_series):
    new_array = []
    for i in range(len(pandas_series)):
        element = pandas_series.iloc[i]

        if len(np.shape(element)) == 0:
            new_array.append(element)
        else:
            new_array.extend(element)

    return np.array(new_array)

def draw_reward_zone():
    for stripe in range(8):
        if stripe % 2 == 0:
            plt.axvline(91.25+stripe*2.5, color='limegreen', linewidth=5.5, alpha=0.4, zorder=0)
        else:
            plt.axvline(91.25+stripe*2.5, color='k', linewidth=5.5, alpha=0.4, zorder=0)


def draw_black_boxes():
    plt.axvline(15, color='k', linewidth=66, alpha=0.25, zorder=0)
    plt.axvline(185, color='k', linewidth=66, alpha=0.25, zorder=0)


def style_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return plt, ax


def style_open_field_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    ax.set_aspect('equal')
    return ax


def style_polar_plot(ax):
    ax.spines['polar'].set_visible(False)
    #ax.grid(None)  
    ax.axvline(math.radians(90), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(180), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(270), color='black', linewidth=1, alpha=0.6)
    ax.axvline(math.radians(0), color='black', linewidth=1, alpha=0.6)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    ax.set_yticklabels([])  # remove yticklabels
    ax.set_xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.xaxis.set_tick_params(labelsize=25)
    return ax


def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]


def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])


def generate_new_color(existing_colors, pastel_factor=0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def style_vr_plot_offset(ax, x_max):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off

    #ax.set_aspect('equal')

    plt.ylim(0, x_max)

    return ax

def pandas_collumn_to_2d_numpy_array(pandas_series):
    new_array = []
    for i in range(len(pandas_series)):
        element = pandas_series.iloc[i]

        if len(np.shape(element)) == 0:
            new_array.append([element])
        else:
            new_array.append(element)

    return np.array(new_array)


def style_vr_plot(ax, x_max=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False, 
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off

    #ax.axvline(0, linewidth=2.5, color='black') # bold line on the y axis
    #ax.axhline(0, linewidth=2.5, color='black') # bold line on the x axis
    if x_max is not None:
        plt.ylim(0, x_max)
    return ax

def style_track_plot(ax, track_length, alpha=0.25):
    ax.axvspan(track_length-60-30-20, track_length-60-30, facecolor='DarkGreen', alpha=alpha, linewidth =0)
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=alpha)# black box

def style_track_plot_no_cue(ax, track_length, alpha=0.25):
    ax.axvline(track_length-60-30-20, color="black", linestyle="dotted", linewidth=1, alpha=alpha)
    ax.axvline(track_length-60-30, color="black", linestyle="dotted", linewidth=1, alpha=alpha)
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=alpha)# black box

    
def style_track_plot_only_black_boxes(ax, track_length, alpha=0.25):
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=alpha)# black box
 

def style_track_plot_multicontext(ax, track_length, alpha=0.25):
    ax.axvspan(90, 110, facecolor='DarkGreen', alpha=alpha, linewidth =0)
    ax.axvspan(120, 140, facecolor='orange', alpha=alpha, linewidth =0)
    ax.axvspan(0, 30, facecolor='k', linewidth =0, alpha=.25) # black box
    ax.axvspan(track_length-30, track_length, facecolor='k', linewidth =0, alpha=alpha)# black box
    

def format_bar_chart(ax, x_label, y_label):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(x_label, fontsize=25)
    ax.set_ylabel(y_label, fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def plot_cumulative_histogram(corr_values, ax, color='black', number_of_bins=40):
    plt.xlim(-1, 1)
    plt.yticks([0, 1])
    ax = format_bar_chart(ax, 'r', 'Cumulative probability')
    values, base = np.histogram(corr_values, bins=number_of_bins, range=(-1, 1))
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(corr_values))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c=color, linewidth=5, alpha=0.6)
    return ax
