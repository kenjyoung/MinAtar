################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
#                                                                                                              #
# python3 plot_return.py -f <directory/file name prefix> -w <window size> -s <granularity> -n <number of runs> #
################################################################################################################

import torch
import numpy, argparse, math
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


################################################################################################################
# process_data
#
# Combines all the runs in different files into a list.  Expects file names of the form 
# <file_name>_<number>_data_and_weights, where <number> loops over every number from 1 to <num_runs>+1. Each 
# element of this list corresponds to a run.  Each run is a list of average returns, our measure of performance.  
# Since the frame that corresponds to when an episode terminates differ from episode to episode and from run to 
# run, we need to sync the data by frames because the total number of frames is the same across the runs.   
# First, we acquire a list of unique frames corresponding to when each episode terminates across all runs. 
# For each run, we average the performance measure over a window size and noted the frame at each cutoff.  Then, 
# we fill the averages into the unique frames up to each frame cutoff.  This ensures that each unique frame 
# contains the same number of averages, so that we can plot the mean measure across the runs with error bars.  
# The function saves the synced data into a file under the same directory as the input files.
#
# Input:
#   num_runs: number of runs given by the user
#   file_name: directory of data files plus the file name prefix (e.g., directory/seaquest)
#   window_size: specify the size of the moving window for averaging the average returns
#
################################################################################################################
def process_data(num_runs, file_name, window_size):

    # Aggregate all the runs from different files
    returns_runs_episodes = []
    frame_stamps_runs_episodes = []
    for i in numpy.arange(1, num_runs+1):
        data_and_weights = torch.load(file_name + "_" + str(i) + "_data_and_weights",
                                      map_location=lambda storage, loc: storage)

        returns = data_and_weights['returns']
        frame_stamps = data_and_weights['frame_stamps']

        returns_runs_episodes.append(returns)
        frame_stamps_runs_episodes.append(frame_stamps)

    # Acquire unique frames across all runs
    flatten_frame_stamps = [item for sublist in frame_stamps_runs_episodes for item in sublist]
    flatten_frame_stamps.sort()
    unique_frame_stamps = numpy.unique(flatten_frame_stamps).tolist()

    # Get smoothed performance measures
    mv_avg = []
    mv_frames = []
    for _, val in enumerate(zip(returns_runs_episodes, frame_stamps_runs_episodes)):
        mv_avg_per_run = numpy.convolve(val[0], numpy.ones((window_size,)) / window_size, mode='valid')
        mv_frames_per_run = [val[1][i+window_size-1] for i, frame
                             in enumerate(numpy.arange(len(val[1]) - window_size + 1))]

        mv_avg.append(mv_avg_per_run)
        mv_frames.append(mv_frames_per_run)

    run_index = [0]*num_runs
    returns_by_frame = numpy.zeros((len(unique_frame_stamps), num_runs))

    # Fill measure into unique frames for each run
    for index, bucket in enumerate(unique_frame_stamps):
        for run in numpy.arange(num_runs):
            if len(mv_frames[run])-1 > run_index[run] and mv_frames[run][run_index[run]] == bucket:
                run_index[run] += 1
            returns_by_frame[index][run] = mv_avg[run][run_index[run]]

    # Save the processed data into a file
    torch.save({
        'returns': returns_by_frame,
        'unique_frames': unique_frame_stamps
    }, file_name+"_processed_data")


################################################################################################################
# plot_avg_return
#
# Plots the processed data according to function process_data.  This function saves the plot in the same
# directory as the processed data.
#
# Inputs:
#   file_name: the directory of the processed data plus the file name prefix (e.g., directory/seaquest)
#   granularity: to avoid overplotting, we sample the plotting point according to this parameter
#
################################################################################################################
def plot_avg_return(file_name, granularity):
    plotting_data = torch.load(file_name + "_processed_data")

    returns = plotting_data['returns']
    unique_frames = plotting_data['unique_frames']
    x_len = len(unique_frames)
    x_index = [i for i in numpy.arange(0, x_len, granularity)]

    x = unique_frames[::granularity]
    y = numpy.transpose(numpy.array(returns)[x_index, :])

    f, ax = plt.subplots(1, 1, figsize=[3, 2], dpi=300)
    sns.set_style("ticks")
    sns.set_context("paper")

    # Find the order of magnitude of the last frame
    order = int(math.log10(unique_frames[-1]))
    range_frames = int(unique_frames[-1]/ (10**order))

    sns.tsplot(data=y, time=numpy.array(x)/(10**order), color='b')
    ax.set_xticks(numpy.arange(range_frames + 1))
    plt.show()

    f.savefig(file_name + "_avg_return.pdf", bbox_inches="tight")
    plt.close(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", type=str)
    parser.add_argument("--windowsize", "-w", type=str)
    parser.add_argument("--granularity", "-s", type=str)
    parser.add_argument("--numruns", "-n", type=str)
    args = parser.parse_args()

    file_path = Path(args.filename + "_processed_data")
    if not file_path.is_file():
        process_data(int(args.numruns), args.filename, int(args.windowsize))

    plot_avg_return(args.filename, int(args.granularity))


if __name__ == '__main__':
    main()
