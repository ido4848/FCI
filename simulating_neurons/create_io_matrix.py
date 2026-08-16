"""
Create the input-output (io) matrix of a neuron model.

The io matrix answers "which input firing rates make this neuron fire at ~1 Hz?". It probes a
grid of (excitatory, inhibitory) input firing rate bins - in spikes per synapse per second - by
running a handful of short simulations per bin, and keeps the bins whose average output firing
rate lands inside `--legitimate_output_fr_range`. Those *legitimate options* are then handed to
`submit_simulate_neuron_and_create_dataset.py`, which draws the input rates of every simulation
in the dataset from them.

This step is model specific and it matters: a model with a large dendritic membrane area needs
very different input rates than a small one to fire at the same rate, so datasets generated with
the wrong rates are not comparable across models and the resulting FCI values are meaningless.

Every model in this repository already ships the io matrix that was used in the paper under
`<neuron_model_folder>/io_matrix/`, so this script is only needed for a new model (or to redo the
calibration with different settings).

Usage:
    python simulating_neurons/create_io_matrix.py \
        --neuron_model_folder <path to model> \
        --io_matrix_folder <output folder> \
        --use_local
"""

from __future__ import print_function
import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import time
import argparse
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.utils import setup_logger, str2bool, ArgumentSaver, AddOutFileAction, AddDefaultInformationAction, TeeAll
from utils.slurm_job import get_job_submitter_args, get_job_args
from simulating_neurons.io_matrix_utils import (get_io_matrix_folder_for_neuron_model_folder,
                                                io_matrix_rows_from_df, save_io_matrix_csv,
                                                save_legitimate_options)
from submit_simulate_neuron_and_create_dataset import get_generate_dataset_args, generate_dataset

import logging
logger = logging.getLogger(__name__)


def get_generate_io_matrix_args():
    saver = ArgumentSaver()

    saver.add_argument('--count_simulations_for_option', default=20, type=int)

    saver.add_argument('--exc_fr_per_synapse_option_range', nargs='+', type=float, default=[0, 20], action=AddDefaultInformationAction)
    saver.add_argument('--exc_fr_per_synapse_bin_size', default=0.05, type=float)
    saver.add_argument('--exc_fr_per_synapse_bin_overlap', default=0.0, type=float)

    saver.add_argument('--inh_fr_per_synapse_option_range', nargs='+', type=float, default=[0, 20], action=AddDefaultInformationAction)
    saver.add_argument('--inh_fr_per_synapse_bin_size', default=0.05, type=float)
    saver.add_argument('--inh_fr_per_synapse_bin_overlap', default=0.0, type=float)

    saver.add_argument('--legitimate_output_fr_range', nargs='+', type=float, default=[0.9, 1.1], action=AddDefaultInformationAction)
    saver.add_argument('--legitimate_options_count', default=20, type=int, action=AddDefaultInformationAction)

    saver.add_argument('--dynamic_search', type=str2bool, nargs='?', const=True, default=True)

    # relevant only when dynamic_search is False
    saver.add_argument('--search_exc_fr_per_synapse_range', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--search_inh_fr_per_synapse_range', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--search_inh_fr_per_synapse_range_inh_overlap_size', default=0.5, type=float)

    saver.add_argument('--allow_exc_bigger_than_inh', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--allow_reuse_exc_range_or_inh_range', type=str2bool, nargs='?', const=True, default=False)

    saver.add_argument('--count_options_per_axis', default=None, type=int,
                       help='Size of the probed grid per iteration. Defaults to what fits in --maximum_cpu_parallel_jobs.')

    dataset_saver = get_generate_dataset_args()
    dataset_saver.add_to_parser(saver)

    return saver

def get_generate_io_matrix_parser():
    parser = argparse.ArgumentParser(description='Generate io_matrix for a neuron')
    parser.add_argument('--neuron_model_folder')
    parser.add_argument('--io_matrix_folder', action=AddOutFileAction)
    parser.add_argument('--io_matrix_name', default=None)

    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--save_to_neuron_model_folder', type=str2bool, nargs='?', const=True, default=False,
                        help='Also write legitimate_options.json and io_matrix.csv into '
                             '<neuron_model_folder>/io_matrix/, where dataset generation picks them up.')

    saver = get_generate_io_matrix_args()
    saver.add_to_parser(parser)

    job_saver = get_job_args()
    job_saver.add_to_parser(parser)

    job_submitter_saver = get_job_submitter_args()
    job_submitter_saver.add_to_parser(parser)

    return parser

def get_args():
    parser = get_generate_io_matrix_parser()
    return parser.parse_args()

def build_ranges(range_limits, bin_size, bin_overlap, count_ranges, start=None):
    """The list of [low, high] bins tiling `range_limits` with the given bin size and overlap."""
    start = range_limits[0] if start is None else start
    starts = start + np.arange(0, count_ranges) * (bin_size - bin_overlap)
    ends = starts + bin_size

    ind = np.argmax(ends > range_limits[1])
    if ind != 0:
        starts = starts[:ind]
        ends = ends[:ind]
    if ends[0] >= range_limits[1]:
        starts = [starts[0]]
        ends = [ends[0]]

    return [(round(low, 3), round(high, 3)) for low, high in zip(starts, ends)]

def plot_io_matrix(args, df, io_matrix_name=None, io_matrix_path=None):
    exc_axis_length = len(set(df.exc_range))
    inh_axis_length = len(set(df.inh_range))

    fr_matrix = np.zeros((exc_axis_length, inh_axis_length)) - 1
    legitimate_options = []
    option_to_xy = {}

    for i, exc_range in enumerate(sorted(set(df.exc_range))):
        for j, inh_range in enumerate(sorted(set(df.inh_range))):
            df_temp = df[(df.inh_range == inh_range) & (df.exc_range == exc_range)]
            if len(df_temp) == 1:
                row = df_temp.iloc[0]
                fr_matrix[i, j] = row.average_output_fr
                option_to_xy[row.option] = (j, i)
                if row.legitimate:
                    legitimate_options.append(row.option)

    fig = plt.figure(figsize=(30, 10))
    ax = fig.subplots(1, 1)

    def highlight_cell(x,y, ax=None, **kwargs):
        rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    im = ax.matshow(fr_matrix, aspect='auto')
    ax.set_xlabel("inh_fr_per_synapse")
    ax.set_ylabel("exc_fr_per_synapse")

    ax.set_yticks(range(0, exc_axis_length))
    ax.set_yticklabels(sorted(set(df.exc_range)))
    ax.yaxis.set_ticks_position("left")

    ax.set_xticks(range(0, inh_axis_length))
    ax.set_xticklabels(sorted(set(df.inh_range)), fontsize=7)
    ax.xaxis.set_ticks_position("bottom")

    for i in range(fr_matrix.shape[0]):
        for j in range(fr_matrix.shape[1]):
            if fr_matrix[i, j] >= 0:
                ax.text(j, i, f'{fr_matrix[i, j]:0.2f}', va='center', ha='center', color='black', weight='bold')

    cax = fig.add_axes([0.27, 0.9, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')

    for option in legitimate_options:
        x, y = option_to_xy[option]
        highlight_cell(x, y, ax=ax, color='green', linewidth=3)

    plt.subplots_adjust(top=0.8)
    suptitle = io_matrix_name if io_matrix_name is not None else os.path.basename(io_matrix_path)
    fig.suptitle(suptitle, y=1.05)

    io_matrix_path = io_matrix_path if io_matrix_path is not None else f"{args.io_matrix_folder}/{io_matrix_name}"

    png_path = f"{io_matrix_path}.png"
    plt.savefig(png_path)
    plt.close('all')

def check_options(args, options, check_name, plot=True, generate_dataset_function=generate_dataset):
    """Run `--count_simulations_for_option` simulations for each (exc_range, inh_range) option."""
    check_folder = os.path.join(os.path.join(args.io_matrix_folder, check_name))

    generate_dataset_args = copy.deepcopy(args)
    generate_dataset_args.simulation_dataset_folder = os.path.join(check_folder, "simulation_dataset")
    generate_dataset_args.simulation_dataset_name = f"{check_name}_simulation_dataset"

    # only the valid set is used for the io matrix, so the other two are kept as small as possible
    generate_dataset_args.count_simulations_for_train = min(10, args.count_simulations_for_train)
    generate_dataset_args.count_simulations_for_test = min(10, args.count_simulations_for_test)

    generate_dataset_args.exc_fr_per_synapse_options = list(np.concatenate([item[0] for item in options]))
    generate_dataset_args.inh_fr_per_synapse_options = list(np.concatenate([item[1] for item in options]))
    # the options being probed are the point of the run, never fall back to the ones of the model
    generate_dataset_args.exc_fr_per_synapse_options_nondefault = True
    generate_dataset_args.inh_fr_per_synapse_options_nondefault = True
    generate_dataset_args.fr_option_choice_by_modulo = True
    generate_dataset_args.count_simulations_for_valid = generate_dataset_args.count_simulations_for_option * len(options)

    check_options_total_duration_in_seconds = generate_dataset_function(generate_dataset_args)

    valid_summary = pickle.load(open(f'{generate_dataset_args.simulation_dataset_folder}/valid/summary.pkl','rb'))

    firing_rates_for_option = {k:[] for k in range(len(options))}
    for i in range(generate_dataset_args.count_simulations_for_valid):
        firing_rates_for_option[i%len(options)].append(valid_summary['firing_rate_information']['by_index'][i])

    rows = []
    for k, output_fr in firing_rates_for_option.items():
        option = options[k]
        average_output_fr = np.mean(output_fr)
        logger.info(f"option {option} has average firing rate of {average_output_fr}")
        rows.append({'k': k, 'option': option, 'exc_range': option[0], 'inh_range': option[1],
                     'output_fr': output_fr, 'average_output_fr': average_output_fr,
                     'std_output_fr': np.std(output_fr), 'legitimate': False})

    df = pd.DataFrame(rows, columns=['k', 'option', 'exc_range', 'inh_range', 'output_fr',
                                     'average_output_fr', 'std_output_fr', 'legitimate'])

    if plot and args.save_plots:
        plot_io_matrix(args, df, io_matrix_name=f"{check_name}_io_matrix", io_matrix_path=f"{check_folder}/{check_name}_io_matrix")

    return check_options_total_duration_in_seconds, df

def calculate_legitimate_options(args, options, df):
    """Pick the options whose average output firing rate is inside --legitimate_output_fr_range."""
    legitimate_option_indices = []
    exc_range_used = set()
    inh_range_used = set()
    for exc_range in sorted(set(df.exc_range)):
        for inh_range in sorted(set(df.inh_range)):
            df_temp = df[(df.inh_range == inh_range) & (df.exc_range == exc_range)]
            if len(df_temp) == 1:
                row = df_temp.iloc[0]
                average_output_fr = row.average_output_fr
                k = row.k
                option = row.option

                if args.legitimate_output_fr_range[0] <= average_output_fr and average_output_fr <= args.legitimate_output_fr_range[1]:
                    if args.allow_exc_bigger_than_inh or option[1][1] >= option[0][1]:
                        if exc_range in exc_range_used or inh_range in inh_range_used:
                            if not args.allow_reuse_exc_range_or_inh_range:
                                logger.info(f"skipping option {option} because exc_range {exc_range} or inh_range {inh_range} is already used")
                                continue
                        logger.info(f"adding option {option} with exc_range {exc_range} and inh_range {inh_range} to legitimate options")
                        exc_range_used.add(exc_range)
                        inh_range_used.add(inh_range)
                        legitimate_option_indices.append(k)
                        df.loc[(df.inh_range == inh_range) & (df.exc_range == exc_range),'legitimate'] = True
                        if len(legitimate_option_indices) == args.legitimate_options_count:
                            break

        if len(legitimate_option_indices) == args.legitimate_options_count:
            break

    legitimate_options = [(options[k][0], options[k][1]) for k in legitimate_option_indices]
    return legitimate_option_indices, legitimate_options, df

def search_exc_fr_per_synapse_range(args, generate_dataset_function=generate_dataset):
    """Find the lowest excitatory bin that, with the lowest inhibitory bin, already fires enough."""
    search_name = "search_exc_fr_bin"

    exc_range = args.exc_fr_per_synapse_option_range
    exc_bin_size = args.exc_fr_per_synapse_bin_size
    exc_bin_overlap = args.exc_fr_per_synapse_bin_overlap
    count_exc_ranges = int(np.ceil((exc_range[1] - exc_range[0]) / (exc_bin_size - exc_bin_overlap)))
    exc_options = build_ranges(exc_range, exc_bin_size, exc_bin_overlap, count_exc_ranges)

    logger.info("exc_options are:")
    logger.info(f"{exc_options}")

    inh_range = args.inh_fr_per_synapse_option_range
    inh_bin_size = args.inh_fr_per_synapse_bin_size
    inh_bin_overlap = args.inh_fr_per_synapse_bin_overlap
    # Specific to this search: only the lowest inhibitory bin
    inh_options = build_ranges(inh_range, inh_bin_size, inh_bin_overlap, 1)

    logger.info("inh_options are:")
    logger.info(f"{inh_options}")

    options = [(exc_option, inh_option) for exc_option in exc_options for inh_option in inh_options]

    search_total_duration_in_seconds, df = check_options(args, options, f"{search_name}_check", generate_dataset_function=generate_dataset_function)

    exc_bin_option_index = None
    for exc_range in sorted(set(df.exc_range)):
        for inh_range in sorted(set(df.inh_range)):
            df_temp = df[(df.inh_range == inh_range) & (df.exc_range == exc_range)]
            if len(df_temp) == 1:
                row = df_temp.iloc[0]
                average_output_fr = row.average_output_fr
                k = row.k

                if average_output_fr >= args.legitimate_output_fr_range[0]:
                    exc_bin_option_index = k
                    break
        if exc_bin_option_index is not None:
            break

    if exc_bin_option_index is None:
        logger.error("No option with sufficient firing rate found")
        raise Exception("No option with sufficient firing rate found")

    exc_bin_range = options[exc_bin_option_index][0]
    inh_bin_range = options[exc_bin_option_index][1]
    logger.info(f"Found exc_bin (/exc_range) {exc_bin_range} that with inh {inh_bin_range} gives sufficient firing rate")

    search_summary = {'args':args, 'options': options, 'exc_bin_option_index':exc_bin_option_index, 'df': df}
    pickle.dump(search_summary, open(f'{args.io_matrix_folder}/{search_name}_summary.pkl', 'wb'), protocol=-1)

    return search_total_duration_in_seconds, exc_bin_range

def search_inh_fr_per_synapse_range(args, generate_dataset_function=generate_dataset):
    """Find the inhibitory range that brings the output firing rate down through the target range."""
    search_name = "search_inh_fr_range"

    exc_bin_size = args.exc_fr_per_synapse_bin_size
    exc_bin_overlap = args.exc_fr_per_synapse_bin_overlap

    exc_range = args.found_exc_fr_per_synapse_option_range
    count_exc_ranges = int(np.ceil((exc_range[1] - exc_range[0]) / (exc_bin_size - exc_bin_overlap)))
    exc_options = build_ranges(exc_range, exc_bin_size, exc_bin_overlap, count_exc_ranges)

    logger.info("exc_options are:")
    logger.info(f"{exc_options}")

    inh_range = args.inh_fr_per_synapse_option_range
    inh_bin_size = args.inh_fr_per_synapse_bin_size
    # Specific to this search
    inh_bin_overlap = args.search_inh_fr_per_synapse_range_inh_overlap_size
    count_inh_ranges = int(np.ceil((inh_range[1] - inh_range[0]) / (inh_bin_size - inh_bin_overlap)))
    inh_options = build_ranges(inh_range, inh_bin_size, inh_bin_overlap, count_inh_ranges)

    logger.info("inh_options are:")
    logger.info(f"{inh_options}")

    options = [(exc_option, inh_option) for exc_option in exc_options for inh_option in inh_options]

    search_total_duration_in_seconds, df = check_options(args, options, f"{search_name}_check", generate_dataset_function=generate_dataset_function)

    exc_bin_option_index = None
    min_range_option_index = None
    max_range_option_index = None
    for exc_range in sorted(set(df.exc_range)):
        for inh_range in sorted(set(df.inh_range)):
            df_temp = df[(df.inh_range == inh_range) & (df.exc_range == exc_range)]
            if len(df_temp) == 1:
                row = df_temp.iloc[0]
                average_output_fr = row.average_output_fr
                k = row.k

                if average_output_fr <= args.legitimate_output_fr_range[1] and min_range_option_index is None:
                    min_range_option_index = k
                    if average_output_fr <= args.legitimate_output_fr_range[0]:
                        if min_range_option_index != 0:
                            min_range_option_index -= 1

                if average_output_fr <= args.legitimate_output_fr_range[0] and max_range_option_index is None:
                    max_range_option_index = k

    if min_range_option_index is None or max_range_option_index is None:
        logger.error("No options with sufficient firing rate found")
        raise Exception("No options with sufficient firing rate found")

    min_exc_bin_range = options[min_range_option_index][0]
    min_inh_bin_range = options[min_range_option_index][1]
    logger.info(f"Found min_inh_bin_range {min_inh_bin_range} that with exc {min_exc_bin_range} gives sufficient firing rate")

    max_exc_bin_range = options[max_range_option_index][0]
    max_inh_bin_range = options[max_range_option_index][1]
    logger.info(f"Found max_inh_bin_range {max_inh_bin_range} that with exc {max_exc_bin_range} gives sufficient firing rate")

    inh_range = [min_inh_bin_range[0], max_inh_bin_range[1]]
    logger.info(f"Found inh_range is {inh_range}")

    search_summary = {'args':args, 'options': options, 'exc_bin_option_index':exc_bin_option_index, 'df': df}
    pickle.dump(search_summary, open(f'{args.io_matrix_folder}/{search_name}_summary.pkl', 'wb'), protocol=-1)

    return search_total_duration_in_seconds, inh_range

def generate_io_matrix_dynamic(args, generate_dataset_function=generate_dataset):
    """
    Probe a small grid of (exc, inh) bins, then keep walking it towards higher rates.

    Each iteration extends the grid along the excitatory axis (as long as the neuron still fires
    too little) and along the inhibitory axis (as long as it still fires too much), so only the
    part of the plane that can produce the target output firing rate is ever simulated.
    """
    if args.count_options_per_axis is not None:
        count_ranges_per_axis = args.count_options_per_axis
    else:
        # leave room for the jobs of the train/test sets of each check
        count_total_cpu_jobs = args.maximum_cpu_parallel_jobs - 10 - 10
        count_total_options = int(count_total_cpu_jobs / args.count_simulations_for_option)
        count_ranges_per_axis = int(np.sqrt(count_total_options))
    count_ranges_per_axis = max(1, count_ranges_per_axis)
    logger.info(f"probing {count_ranges_per_axis}x{count_ranges_per_axis} options per iteration")

    exc_range = args.exc_fr_per_synapse_option_range
    exc_bin_size = args.exc_fr_per_synapse_bin_size
    exc_bin_overlap = args.exc_fr_per_synapse_bin_overlap
    initial_exc_ranges = build_ranges(exc_range, exc_bin_size, exc_bin_overlap, count_ranges_per_axis)

    inh_range = args.inh_fr_per_synapse_option_range
    inh_bin_size = args.inh_fr_per_synapse_bin_size
    inh_bin_overlap = args.inh_fr_per_synapse_bin_overlap
    initial_inh_ranges = build_ranges(inh_range, inh_bin_size, inh_bin_overlap, count_ranges_per_axis)

    max_input_exc = args.exc_fr_per_synapse_option_range[1]
    max_input_inh = args.inh_fr_per_synapse_option_range[1]

    output_fr_range = args.legitimate_output_fr_range

    ranges_to_explore = {}
    for exc_range in initial_exc_ranges:
        ranges_to_explore[exc_range] = initial_inh_ranges

    search_time = 0
    iteration_count = 0

    df = None

    while len(ranges_to_explore) > 0:
        options_to_check = []
        for exc_range in sorted(list(ranges_to_explore.keys())):
            for inh_range in sorted(ranges_to_explore[exc_range]):
                options_to_check.append((exc_range, inh_range))

        logger.info("options_to_check are:")
        logger.info(f"{options_to_check}")

        current_search_time, cur_df = check_options(args, options_to_check, f"dynamic_io_matrix_iteration_{iteration_count}_check", generate_dataset_function=generate_dataset_function)

        search_time += current_search_time
        iteration_count += 1

        if df is None:
            df = cur_df
        else:
            df = pd.concat([df, cur_df])

        df['k'] = range(0, len(df))

        options = df['option'].tolist()

        io_matrix_name = f"dynamic_io_matrix_after_{iteration_count}_iterations"
        io_matrix_path = f"{args.io_matrix_folder}/{io_matrix_name}"

        if args.save_plots:
            plot_io_matrix(args, df, io_matrix_name=io_matrix_name, io_matrix_path=io_matrix_path)

        df.to_csv(f"{io_matrix_path}.csv", index=False)

        legitimate_option_indices, legitimate_options, df = calculate_legitimate_options(args, options, df)

        # found enough, breaking
        if len(legitimate_option_indices) == args.legitimate_options_count:
            break

        new_ranges_to_explore = {}
        for exc_range in sorted(list(ranges_to_explore.keys())):
            inh_range_to_average_output_fr = {}
            for inh_range in sorted(ranges_to_explore[exc_range]):
                df_temp = df[(df.inh_range == inh_range) & (df.exc_range == exc_range)]
                if len(df_temp) == 1:
                    row = df_temp.iloc[0]
                    average_output_fr = row.average_output_fr
                    inh_range_to_average_output_fr[inh_range] = average_output_fr

            # last inh_range
            if average_output_fr > output_fr_range[0] and inh_range[1] < max_input_inh:
                new_ranges_to_explore[exc_range] = [(round(inh_range[0]+(inh_bin_size-inh_bin_overlap)*i,3), round(inh_range[1]+(inh_bin_size-inh_bin_overlap)*i,3)) for i in range(1, count_ranges_per_axis + 1)]

        # last exc_range
        not_too_low_inh_ranges = [inh_range for inh_range, average_output_fr in inh_range_to_average_output_fr.items() if average_output_fr < output_fr_range[1]]
        if len(not_too_low_inh_ranges) > 0 and exc_range[1] < max_input_exc:
            new_ranges_to_explore[(round(exc_range[0]+(exc_bin_size-exc_bin_overlap),3), round(exc_range[1]+(exc_bin_size-exc_bin_overlap),3))] = not_too_low_inh_ranges

        ranges_to_explore = new_ranges_to_explore

    return search_time, options, legitimate_option_indices, legitimate_options, df

def save_io_matrix_files(args, legitimate_options, df, io_matrix_folder):
    """Write legitimate_options.json and io_matrix.csv, the two files a model folder ships."""
    legitimate_df = df[df.legitimate == True]
    average_output_frs = []
    std_output_frs = []
    for exc_range, inh_range in legitimate_options:
        row = legitimate_df[(legitimate_df.exc_range == exc_range) & (legitimate_df.inh_range == inh_range)].iloc[0]
        average_output_frs.append(row.average_output_fr)
        std_output_frs.append(row.std_output_fr)

    calibration = {
        "legitimate_output_fr_range": list(args.legitimate_output_fr_range),
        "count_simulations_for_option": args.count_simulations_for_option,
        "simulation_duration_in_seconds": args.simulation_duration_in_seconds,
        "exc_fr_per_synapse_bin_size": args.exc_fr_per_synapse_bin_size,
        "inh_fr_per_synapse_bin_size": args.inh_fr_per_synapse_bin_size,
        "dynamic_search": args.dynamic_search,
        "neuron_model_folder": args.neuron_model_folder,
    }

    save_legitimate_options(io_matrix_folder, legitimate_options, calibration=calibration,
                            average_output_frs=average_output_frs, std_output_frs=std_output_frs)
    save_io_matrix_csv(io_matrix_folder, io_matrix_rows_from_df(df))

def generate_io_matrix(args, generate_dataset_function=generate_dataset):
    logger.info("Going to generate io_matrix with args:")
    logger.info("{}".format(args))
    logger.info("...")

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)

    io_matrix_basename = args.io_matrix_name
    if not io_matrix_basename:
        io_matrix_basename = os.path.basename(args.io_matrix_folder)

    args.io_matrix_basename = io_matrix_basename

    os.makedirs(args.io_matrix_folder, exist_ok=True)

    if args.dynamic_search:
        generate_io_matrix_total_duration_in_seconds, options, legitimate_option_indices, legitimate_options, df = generate_io_matrix_dynamic(args, generate_dataset_function=generate_dataset_function)
    else:
        search_time = 0
        exc_range = args.exc_fr_per_synapse_option_range
        if args.search_exc_fr_per_synapse_range:
            search_time, exc_range = search_exc_fr_per_synapse_range(args, generate_dataset_function=generate_dataset_function)
        args.found_exc_fr_per_synapse_option_range = exc_range

        exc_bin_size = args.exc_fr_per_synapse_bin_size
        exc_bin_overlap = args.exc_fr_per_synapse_bin_overlap
        count_exc_ranges = int(np.ceil((exc_range[1] - exc_range[0]) / (exc_bin_size - exc_bin_overlap)))
        exc_options = build_ranges(exc_range, exc_bin_size, exc_bin_overlap, count_exc_ranges)
        args.exc_options = exc_options

        logger.info("exc_options are:")
        logger.info(f"{exc_options}")

        inh_range = args.inh_fr_per_synapse_option_range
        if args.search_inh_fr_per_synapse_range:
            inh_search_time, inh_range = search_inh_fr_per_synapse_range(args, generate_dataset_function=generate_dataset_function)
            search_time += inh_search_time
        args.found_inh_fr_per_synapse_option_range = inh_range

        inh_bin_size = args.inh_fr_per_synapse_bin_size
        inh_bin_overlap = args.inh_fr_per_synapse_bin_overlap
        count_inh_ranges = int(np.ceil((inh_range[1] - inh_range[0]) / (inh_bin_size - inh_bin_overlap)))
        inh_options = build_ranges(inh_range, inh_bin_size, inh_bin_overlap, count_inh_ranges)
        args.inh_options = inh_options

        logger.info("inh_options are:")
        logger.info(f"{inh_options}")

        options = [(exc_option, inh_option) for exc_option in exc_options for inh_option in inh_options]

        final_check_total_duration_in_second, df = check_options(args, options, "io_matrix_check", plot=False, generate_dataset_function=generate_dataset_function)
        generate_io_matrix_total_duration_in_seconds = search_time + final_check_total_duration_in_second

        legitimate_option_indices, legitimate_options, df = calculate_legitimate_options(args, options, df)

    logger.info(f"legitimate_options are:")
    for legitimate_option in legitimate_options:
        logger.info(f"exc = {legitimate_option[0]}, inh = {legitimate_option[1]}")

    io_matrix_summary = {'args':args, 'options': options, 'legitimate_options':legitimate_options, 'legitimate_option_indices': legitimate_option_indices, 'df': df}
    pickle.dump(io_matrix_summary, open(f'{args.io_matrix_folder}/summary.pkl', 'wb'), protocol=-1)

    if args.save_plots:
        plot_io_matrix(args, df, "io_matrix")

    save_io_matrix_files(args, legitimate_options, df, args.io_matrix_folder)

    if args.save_to_neuron_model_folder:
        model_io_matrix_folder = get_io_matrix_folder_for_neuron_model_folder(args.neuron_model_folder)
        save_io_matrix_files(args, legitimate_options, df, model_io_matrix_folder)
        logger.info(f"io matrix also written to the model folder, {model_io_matrix_folder}")

    if len(legitimate_options) != args.legitimate_options_count:
        raise Exception(f"legitimate_options_count={args.legitimate_options_count} but len(legitimate_options)={len(legitimate_options)}")

    logger.info(f"generate io_matrix finished!, it took {generate_io_matrix_total_duration_in_seconds/60.0:.3f} minutes")

    return generate_io_matrix_total_duration_in_seconds

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())

    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to generate io_matrix for neuron! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    generate_io_matrix(args)
    logger.info(f"Goodbye from generate io_matrix for neuron! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
