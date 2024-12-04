from __future__ import print_function
import pathlib
import importlib
import copy
import os
import peakutils
import h5py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
import pickle
import time
import argparse
from scipy import sparse

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils import setup_logger, str2bool, ArgumentSaver, AddOutFileAction, TeeAll, AddDefaultInformationAction

import logging
logger = logging.getLogger(__name__)

# mock
neuron = None
h = None
gui = None

def generate_input_spike_rates_for_simulation(args, sim_duration_ms, count_exc_netcons, count_inh_netcons):
    auxiliary_information = {}

    # randomly sample inst rate (with some uniform noise) smoothing sigma
    keep_inst_rate_const_for_ms = args.inst_rate_sampling_time_interval_options_ms[np.random.randint(len(args.inst_rate_sampling_time_interval_options_ms))]
    keep_inst_rate_const_for_ms += int(2 * args.inst_rate_sampling_time_interval_jitter_range * np.random.rand() - args.inst_rate_sampling_time_interval_jitter_range)
    
    # randomly sample smoothing sigma (with some uniform noise)
    temporal_inst_rate_smoothing_sigma = args.temporal_inst_rate_smoothing_sigma_options_ms[np.random.randint(len(args.temporal_inst_rate_smoothing_sigma_options_ms))]
    temporal_inst_rate_smoothing_sigma += int(2 * args.temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - args.temporal_inst_rate_smoothing_sigma_jitter_range)
    
    count_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))
    
    # create the coarse inst rates with units of "total spikes per tree per 100 ms"
    count_exc_spikes_per_100ms   = np.random.uniform(low=args.effective_count_exc_spikes_per_synapse_per_100ms_range[0] * count_exc_netcons,
     high=args.effective_count_exc_spikes_per_synapse_per_100ms_range[1] * count_exc_netcons, size=(1,count_inst_rate_samples))

    if args.adaptive_inh:
        count_inh_spikes_per_synapse_per_100ms_low_range = np.maximum(0, count_exc_spikes_per_100ms / count_exc_netcons + args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range[0])
        count_inh_spikes_per_synapse_per_100ms_high_range = count_exc_spikes_per_100ms / count_exc_netcons + args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range[1]
        count_inh_spikes_per_100ms = np.random.uniform(low=count_inh_spikes_per_synapse_per_100ms_low_range * count_inh_netcons,
         high=count_inh_spikes_per_synapse_per_100ms_high_range * count_inh_netcons, size=(1,count_inst_rate_samples))
    else:
        count_inh_spikes_per_100ms  = np.random.uniform(low=args.effective_count_inh_spikes_per_synapse_per_100ms_range[0] * count_inh_netcons, high=args.effective_count_inh_spikes_per_synapse_per_100ms_range[1] * count_inh_netcons, size=(1,count_inst_rate_samples))

    # convert to units of "per_netcon_per_1ms"
    exc_spike_rate_per_netcon_per_1ms   = count_exc_spikes_per_100ms   / (count_exc_netcons  * 100.0)
    inh_spike_rate_per_netcon_per_1ms  = count_inh_spikes_per_100ms  / (count_inh_netcons  * 100.0)
            
    # kron by space (uniform distribution across branches per tree)
    exc_spike_rate_per_netcon_per_1ms   = np.kron(exc_spike_rate_per_netcon_per_1ms  , np.ones((count_exc_netcons,1)))
    inh_spike_rate_per_netcon_per_1ms  = np.kron(inh_spike_rate_per_netcon_per_1ms , np.ones((count_inh_netcons,1)))
        
    # vstack basal and apical
    exc_spike_rate_per_netcon_per_1ms  = np.vstack((exc_spike_rate_per_netcon_per_1ms))
    inh_spike_rate_per_netcon_per_1ms = np.vstack((inh_spike_rate_per_netcon_per_1ms))

    exc_spatial_multiplicative_randomness_delta = np.random.uniform(args.exc_spatial_multiplicative_randomness_delta_range[0], args.exc_spatial_multiplicative_randomness_delta_range[1])
    if np.random.rand() < args.same_exc_inh_spatial_multiplicative_randomness_delta_prob:
        inh_spatial_multiplicative_randomness_delta = exc_spatial_multiplicative_randomness_delta
    else:
        inh_spatial_multiplicative_randomness_delta = np.random.uniform(args.inh_spatial_multiplicative_randomness_delta_range[0], args.inh_spatial_multiplicative_randomness_delta_range[1])

    # add some spatial multiplicative randomness (that will be added to the sampling noise)
    if args.spatial_multiplicative_randomness and np.random.rand() < args.exc_spatial_multiplicative_randomness_delta_prob:
        exc_spike_rate_per_netcon_per_1ms  = np.random.uniform(low=1 - exc_spatial_multiplicative_randomness_delta, high=1 + exc_spatial_multiplicative_randomness_delta, size=exc_spike_rate_per_netcon_per_1ms.shape) * exc_spike_rate_per_netcon_per_1ms
    if args.spatial_multiplicative_randomness and np.random.rand() < args.inh_spatial_multiplicative_randomness_delta_prob:
        inh_spike_rate_per_netcon_per_1ms = np.random.uniform(low=1 - inh_spatial_multiplicative_randomness_delta, high=1 + inh_spatial_multiplicative_randomness_delta, size=inh_spike_rate_per_netcon_per_1ms.shape) * inh_spike_rate_per_netcon_per_1ms

    # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
    exc_spike_rate_per_netcon_per_1ms  = np.kron(exc_spike_rate_per_netcon_per_1ms , np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    inh_spike_rate_per_netcon_per_1ms = np.kron(inh_spike_rate_per_netcon_per_1ms, np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    
    # filter the inst rates according to smoothing sigma
    smoothing_window = signal.gaussian(1.0 + args.temporal_inst_rate_smoothing_sigma_mult * temporal_inst_rate_smoothing_sigma, std=temporal_inst_rate_smoothing_sigma)[np.newaxis,:]
    smoothing_window /= smoothing_window.sum()
    netcon_inst_rate_exc_smoothed  = signal.convolve(exc_spike_rate_per_netcon_per_1ms,  smoothing_window, mode='same')
    netcon_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_netcon_per_1ms, smoothing_window, mode='same')

    return netcon_inst_rate_exc_smoothed, netcon_inst_rate_inh_smoothed, auxiliary_information

def sample_spikes_from_rates(args, netcon_inst_rate_ex, netcon_inst_rate_inh):
    # sample the instantanous spike prob and then sample the actual spikes
    exc_inst_spike_prob = np.random.exponential(scale=netcon_inst_rate_ex)
    exc_spikes_bin      = np.random.rand(exc_inst_spike_prob.shape[0], exc_inst_spike_prob.shape[1]) < exc_inst_spike_prob
    
    inh_inst_spike_prob = np.random.exponential(scale=netcon_inst_rate_inh)
    inh_spikes_bin      = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob

    return exc_spikes_bin, inh_spikes_bin

class MoreThanOneEventPerMsException(Exception):
    pass

class ForceNumberOfSegmentsIsNotAMultipleOfNumberOfSegments(Exception):
    pass

def generate_input_spike_trains_for_simulation_new(args, sim_duration_ms, count_exc_netcons, count_inh_netcons):
    auxiliary_information = {}

    inst_rate_exc, inst_rate_inh, original_spike_rates_information = generate_input_spike_rates_for_simulation(args, sim_duration_ms, count_exc_netcons, count_inh_netcons)
    
    auxiliary_information['original_spike_rates_information'] = original_spike_rates_information

    # build the final rates matrices
    inst_rate_exc_final = inst_rate_exc.copy()
    inst_rate_inh_final = inst_rate_inh.copy()

    # correct any minor mistakes
    inst_rate_exc_final[inst_rate_exc_final <= 0] = 0
    inst_rate_inh_final[inst_rate_inh_final <= 0] = 0

    exc_spikes_bin, inh_spikes_bin = sample_spikes_from_rates(args, inst_rate_exc_final, inst_rate_inh_final)

    for spikes_bin in exc_spikes_bin:
        spike_times = np.nonzero(spikes_bin)[0]
        if len(list(spike_times)) != len(set(spike_times)):
            raise MoreThanOneEventPerMsException("there is more than one event per ms!")

    for spikes_bin in inh_spikes_bin:
        spike_times = np.nonzero(spikes_bin)[0]
        if len(list(spike_times)) != len(set(spike_times)):
            raise MoreThanOneEventPerMsException("there is more than one event per ms!")

    return exc_spikes_bin, inh_spikes_bin, auxiliary_information

def generate_spike_times_and_weights_for_kernel_based_weights(args, syns, simulation_duration_in_ms):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    if args.zero_padding_for_initialization:
        raise NotImplementedError("zero_padding_for_initialization is not implemented for kernel based weights")

    auxiliary_information = {}
    
    auxiliary_information['seg_lens'] = syns.seg_lens

    count_exc_initial_synapses_per_super_synapse = np.ceil(syns.seg_lens).astype(int)
    count_inh_initial_synapses_per_super_synapse = np.ceil(syns.seg_lens).astype(int)

    if args.force_count_initial_synapses_per_super_synapse is not None:
        count_exc_initial_synapses_per_super_synapse = np.array([args.force_count_initial_synapses_per_super_synapse for _ in count_exc_initial_synapses_per_super_synapse])
        count_inh_initial_synapses_per_super_synapse = np.array([args.force_count_initial_synapses_per_super_synapse for _ in count_inh_initial_synapses_per_super_synapse])

    if args.force_count_initial_synapses_per_tree is not None:
        average_number_of_initial_synapses_per_super_synapse = args.force_count_initial_synapses_per_tree // len(count_exc_initial_synapses_per_super_synapse)
        auxiliary_information['average_number_of_initial_synapses_per_super_synapse'] = average_number_of_initial_synapses_per_super_synapse
        count_exc_initial_synapses_per_super_synapse = np.array([average_number_of_initial_synapses_per_super_synapse for _ in count_exc_initial_synapses_per_super_synapse])
        for _ in range(args.force_count_initial_synapses_per_tree % len(count_exc_initial_synapses_per_super_synapse)):
            count_exc_initial_synapses_per_super_synapse[np.random.randint(len(count_exc_initial_synapses_per_super_synapse))] += 1
        
        count_inh_initial_synapses_per_super_synapse = count_exc_initial_synapses_per_super_synapse

    auxiliary_information['count_exc_initial_synapses_per_super_synapse'] = count_exc_initial_synapses_per_super_synapse
    auxiliary_information['count_inh_initial_synapses_per_super_synapse'] = count_inh_initial_synapses_per_super_synapse

    original_count_exc_initial_neurons = count_exc_initial_neurons = np.sum(count_exc_initial_synapses_per_super_synapse)
    original_count_inh_initial_neurons = count_inh_initial_neurons = np.sum(count_inh_initial_synapses_per_super_synapse)

    if args.force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length:
        average_segment_length = np.mean(syns.seg_lens)
        args.effective_count_exc_spikes_per_synapse_per_100ms_range = [args.count_exc_spikes_per_synapse_per_100ms_range[0] * average_segment_length, args.count_exc_spikes_per_synapse_per_100ms_range[1] * average_segment_length]
        args.effective_count_inh_spikes_per_synapse_per_100ms_range = [args.count_inh_spikes_per_synapse_per_100ms_range[0] * average_segment_length, args.count_inh_spikes_per_synapse_per_100ms_range[1] * average_segment_length]
        args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range = [args.adaptive_inh_additive_factor_per_synapse_per_100ms_range[0] * average_segment_length, args.adaptive_inh_additive_factor_per_synapse_per_100ms_range[1] * average_segment_length]
    else:
        args.effective_count_exc_spikes_per_synapse_per_100ms_range = args.count_exc_spikes_per_synapse_per_100ms_range
        args.effective_count_inh_spikes_per_synapse_per_100ms_range = args.count_inh_spikes_per_synapse_per_100ms_range
        args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range = args.adaptive_inh_additive_factor_per_synapse_per_100ms_range

    exc_initial_neurons_spikes_bin, inh_initial_neurons_spikes_bin, initial_neurons_aux_info = generate_input_spike_trains_for_simulation_new(args, simulation_duration_in_ms, count_exc_initial_neurons, count_inh_initial_neurons)
    
    auxiliary_information['initial_neurons_spike_trains_information'] = initial_neurons_aux_info
    auxiliary_information['exc_initial_neurons_spikes_bin'] = exc_initial_neurons_spikes_bin
    auxiliary_information['inh_initial_neurons_spikes_bin'] = inh_initial_neurons_spikes_bin

    exc_initial_neurons_spikes_bin = np.array(exc_initial_neurons_spikes_bin)
    inh_initial_neurons_spikes_bin = np.array(inh_initial_neurons_spikes_bin)

    logger.info(f"exc_initial_neurons_spikes_bin.shape is {exc_initial_neurons_spikes_bin.shape}")
    logger.info(f"inh_initial_neurons_spikes_bin.shape is {inh_initial_neurons_spikes_bin.shape}")

    # Done generating input spikes, now to weights and weighted spikes

 
    exc_initial_neuron_connection_counts = None
    exc_super_synapse_kernels = []
    exc_weighted_spikes = np.zeros((len(exc_netcons), simulation_duration_in_ms))
    exc_ncon_to_input_spike_times = {}
    count_exc_spikes = 0
    count_weighted_exc_spikes = 0
    exc_initial_neurons_weights = []
    for exc_netcon_index, exc_netcon in enumerate(exc_netcons):
        relevant_exc_initial_neurons_spikes_bin = exc_initial_neurons_spikes_bin[np.sum(count_exc_initial_synapses_per_super_synapse[:exc_netcon_index]):np.sum(count_exc_initial_synapses_per_super_synapse[:exc_netcon_index+1])]
        exc_super_synapse_random_kernel = np.random.uniform(low=args.exc_weights_ratio_range[0], high=args.exc_weights_ratio_range[1], size=(1, relevant_exc_initial_neurons_spikes_bin.shape[0]))
        exc_super_synapse_kernels.append(exc_super_synapse_random_kernel)
        exc_initial_neurons_weights += list(exc_super_synapse_random_kernel[exc_super_synapse_random_kernel!=0])
        weighted_spikes = np.dot(exc_super_synapse_random_kernel, relevant_exc_initial_neurons_spikes_bin).flatten()
        count_weighted_exc_spikes += np.sum(weighted_spikes)

        exc_weighted_spikes[exc_netcon_index, :] = weighted_spikes
        exc_ncon_to_input_spike_times[exc_netcon] = np.nonzero(weighted_spikes)[0]
        count_exc_spikes += len(exc_ncon_to_input_spike_times[exc_netcon])

    exc_wiring_matrix = np.zeros((exc_super_synapse_kernels[0].shape[1], len(exc_super_synapse_kernels)))

    inh_initial_neuron_connection_counts = None
    inh_super_synapse_kernels = []
    inh_weighted_spikes = np.zeros((len(inh_netcons), simulation_duration_in_ms))
    inh_ncon_to_input_spike_times = {}
    count_inh_spikes = 0
    count_weighted_inh_spikes = 0
    inh_initial_neurons_weights = []
    for inh_netcon_index, inh_netcon in enumerate(inh_netcons):    
        relevant_inh_initial_neurons_spikes_bin = inh_initial_neurons_spikes_bin[np.sum(count_inh_initial_synapses_per_super_synapse[:inh_netcon_index]):np.sum(count_inh_initial_synapses_per_super_synapse[:inh_netcon_index+1])]
        inh_super_synapse_random_kernel = np.random.uniform(low=args.inh_weights_ratio_range[0], high=args.inh_weights_ratio_range[1], size=(1, relevant_inh_initial_neurons_spikes_bin.shape[0]))
        inh_super_synapse_kernels.append(inh_super_synapse_random_kernel)
        inh_initial_neurons_weights += list(inh_super_synapse_random_kernel[inh_super_synapse_random_kernel!=0])
        weighted_spikes = np.dot(inh_super_synapse_random_kernel, relevant_inh_initial_neurons_spikes_bin).flatten()
        count_weighted_inh_spikes += np.sum(weighted_spikes)

        inh_weighted_spikes[inh_netcon_index, :] = weighted_spikes
        inh_ncon_to_input_spike_times[inh_netcon] = np.nonzero(weighted_spikes)[0]
        count_inh_spikes += len(inh_ncon_to_input_spike_times[inh_netcon])

    inh_wiring_matrix = np.zeros((inh_super_synapse_kernels[0].shape[1], len(inh_super_synapse_kernels)))

    auxiliary_information = save_more_auxiliary_information(args, syns, simulation_duration_in_ms, exc_weighted_spikes, inh_weighted_spikes, \
        count_exc_spikes, count_inh_spikes, auxiliary_information, exc_initial_neurons_weights=exc_initial_neurons_weights, \
            inh_initial_neurons_weights=inh_initial_neurons_weights, \
                exc_initial_neuron_connection_counts=exc_initial_neuron_connection_counts, \
                    inh_initial_neuron_connection_counts=inh_initial_neuron_connection_counts, \
                        exc_wiring_matrix=exc_wiring_matrix, inh_wiring_matrix=inh_wiring_matrix)

    return simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information

def generate_spike_times_and_weights_from_input_file(args, syns):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    auxiliary_information = {}
    auxiliary_information["input_file"] = args.input_file

    weighted_spikes = sparse.load_npz(args.input_file).A

    simulation_duration_in_ms = 0
    if args.add_explicit_padding_for_initialization:
        simulation_duration_in_ms = args.simulation_initialization_duration_in_ms
    simulation_duration_in_ms += weighted_spikes.shape[1]

    if args.add_explicit_padding_for_initialization:
        if args.zero_padding_for_initialization:
            zero_exc_weighted_spikes = np.zeros((len(exc_netcons), args.simulation_initialization_duration_in_ms))
            zero_inh_weighted_spikes = np.zeros((len(inh_netcons), args.simulation_initialization_duration_in_ms))

            padding_exc_weighted_spikes = zero_exc_weighted_spikes
            padding_inh_weighted_spikes = zero_inh_weighted_spikes
        else:
            _, _, _, noise_padding_exc_weighted_spikes, noise_padding_inh_weighted_spikes, noise_padding_auxiliary_information = generate_spike_times_and_weights_for_kernel_based_weights(args, syns, args.simulation_initialization_duration_in_ms)
            auxiliary_information['noise_padding_auxiliary_information'] = noise_padding_auxiliary_information

            padding_exc_weighted_spikes = noise_padding_exc_weighted_spikes
            padding_inh_weighted_spikes = noise_padding_inh_weighted_spikes
    else:
        padding_exc_weighted_spikes = None
        padding_inh_weighted_spikes = None

    ret = generate_spike_times_and_weights_from_weighted_spikes(args, syns, simulation_duration_in_ms,
     weighted_spikes, padding_exc_weighted_spikes=padding_exc_weighted_spikes, padding_inh_weighted_spikes=padding_inh_weighted_spikes)
     
    exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, count_exc_spikes, count_inh_spikes = ret

    auxiliary_information = save_more_auxiliary_information(args, syns, simulation_duration_in_ms, exc_weighted_spikes, inh_weighted_spikes, \
        count_exc_spikes, count_inh_spikes, auxiliary_information)

    return simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information

def generate_spike_times_and_weights_from_weighted_spikes(args, syns, simulation_duration_in_ms, weighted_spikes,
 padding_exc_weighted_spikes=None, padding_inh_weighted_spikes=None):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    if weighted_spikes.min() < 0:
        raise ValueError("weighted_spikes contains negative values")        

    exc_weighted_spikes = np.zeros((len(exc_netcons), simulation_duration_in_ms))
    exc_ncon_to_input_spike_times = {}
    count_exc_spikes = 0
    for exc_netcon_index, exc_netcon in enumerate(exc_netcons):
        if padding_exc_weighted_spikes is not None:
            cur_exc_weighted_spikes = np.concatenate((padding_exc_weighted_spikes[exc_netcon_index,:], weighted_spikes[exc_netcon_index,:]))
        else:
            cur_exc_weighted_spikes = weighted_spikes[exc_netcon_index,:]
        exc_weighted_spikes[exc_netcon_index, :] = cur_exc_weighted_spikes
        exc_ncon_to_input_spike_times[exc_netcon] = np.nonzero(cur_exc_weighted_spikes)[0]
        count_exc_spikes += len(exc_ncon_to_input_spike_times[exc_netcon])

    inh_weighted_spikes = np.zeros((len(inh_netcons), simulation_duration_in_ms))
    inh_ncon_to_input_spike_times = {}
    count_inh_spikes = 0
    for inh_netcon_index, inh_netcon in enumerate(inh_netcons):
        if padding_inh_weighted_spikes is not None:
            cur_inh_weighted_spikes = np.concatenate((padding_inh_weighted_spikes[inh_netcon_index,:], weighted_spikes[len(exc_netcons) + inh_netcon_index,:]))
        else:
            cur_inh_weighted_spikes = weighted_spikes[len(exc_netcons) + inh_netcon_index,:]
        inh_weighted_spikes[inh_netcon_index, :] = cur_inh_weighted_spikes
        inh_ncon_to_input_spike_times[inh_netcon] = np.nonzero(cur_inh_weighted_spikes)[0]
        count_inh_spikes += len(inh_ncon_to_input_spike_times[inh_netcon])

    return exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, count_exc_spikes, count_inh_spikes

def save_more_auxiliary_information(args, syns, simulation_duration_in_ms, exc_weighted_spikes, inh_weighted_spikes, \
    count_exc_spikes, count_inh_spikes, auxiliary_information, exc_initial_neurons_weights=None, inh_initial_neurons_weights=None, \
        exc_initial_neuron_connection_counts=None, inh_initial_neuron_connection_counts=None, exc_wiring_matrix=None, inh_wiring_matrix=None):

    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    # exc
    
    average_exc_spikes_per_second = count_exc_spikes / (simulation_duration_in_ms / 1000)
    count_exc_spikes_per_super_synapse = count_exc_spikes / (len(exc_netcons) + 0.0)
    average_exc_spikes_per_super_synapse_per_second = count_exc_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)

    auxiliary_information['count_exc_spikes'] = count_exc_spikes
    auxiliary_information['average_exc_spikes_per_second'] = average_exc_spikes_per_second
    auxiliary_information['count_exc_spikes_per_super_synapse'] = count_exc_spikes_per_super_synapse
    auxiliary_information['average_exc_spikes_per_super_synapse_per_second'] = average_exc_spikes_per_super_synapse_per_second

    logger.info(f'average of exc spikes per second is {average_exc_spikes_per_second:.3f}, which is {average_exc_spikes_per_super_synapse_per_second:.3f} average exc spikes per exc netcon per second')

    count_weighted_exc_spikes = np.sum(exc_weighted_spikes)

    average_weighted_exc_spikes_per_second = count_weighted_exc_spikes / (simulation_duration_in_ms / 1000)
    count_weighted_exc_spikes_per_super_synapse = count_weighted_exc_spikes / (len(exc_netcons) + 0.0)
    average_weighted_exc_spikes_per_super_synapse_per_second = count_weighted_exc_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)
    
    auxiliary_information['count_weighted_exc_spikes'] = count_weighted_exc_spikes
    auxiliary_information['average_weighted_exc_spikes_per_second'] = average_weighted_exc_spikes_per_second
    auxiliary_information['count_weighted_exc_spikes_per_super_synapse'] = count_weighted_exc_spikes_per_super_synapse
    auxiliary_information['average_weighted_exc_spikes_per_super_synapse_per_second'] = average_weighted_exc_spikes_per_super_synapse_per_second

    logger.info(f'average of weighted exc spikes per second is {average_weighted_exc_spikes_per_second:.3f}, which is {average_weighted_exc_spikes_per_super_synapse_per_second:.3f} average weighted exc spikes per exc netcon per second')

    if exc_initial_neurons_weights is None:
        exc_initial_neurons_weights = [0.0]
    average_exc_initial_neuron_weight = np.mean(exc_initial_neurons_weights)
    auxiliary_information['exc_initial_neurons_weights'] = exc_initial_neurons_weights
    auxiliary_information['average_exc_initial_neuron_weight'] = average_exc_initial_neuron_weight

    logger.info(f'average exc initial neuron weight is {average_exc_initial_neuron_weight:.3f}')

    if exc_initial_neuron_connection_counts is not None:
        logger.info(f'min, max, avg, std, med exc initial neuron connection count are {np.min(exc_initial_neuron_connection_counts):.3f}, {np.max(exc_initial_neuron_connection_counts):.3f}, {np.mean(exc_initial_neuron_connection_counts):.3f}, {np.std(exc_initial_neuron_connection_counts):.3f}, {np.median(exc_initial_neuron_connection_counts):.3f}')
        auxiliary_information['exc_initial_neuron_connection_counts'] = exc_initial_neuron_connection_counts

    auxiliary_information['exc_wiring_matrix'] = exc_wiring_matrix

    # inh
   
    average_inh_spikes_per_second = count_inh_spikes / (simulation_duration_in_ms / 1000)
    count_inh_spikes_per_super_synapse = count_inh_spikes / (len(inh_netcons) + 0.0)
    average_inh_spikes_per_super_synapse_per_second = count_inh_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)

    auxiliary_information['count_inh_spikes'] = count_inh_spikes
    auxiliary_information['average_inh_spikes_per_second'] = average_inh_spikes_per_second
    auxiliary_information['count_inh_spikes_per_super_synapse'] = count_inh_spikes_per_super_synapse
    auxiliary_information['average_inh_spikes_per_super_synapse_per_second'] = average_inh_spikes_per_super_synapse_per_second

    logger.info(f'average number of inh spikes per second is {average_inh_spikes_per_second:.3f}, which is {average_inh_spikes_per_super_synapse_per_second:.3f} average inh spikes per inh netcon per second')

    count_weighted_inh_spikes = np.sum(inh_weighted_spikes)

    average_weighted_inh_spikes_per_second = count_weighted_inh_spikes / (simulation_duration_in_ms / 1000)
    count_weighted_inh_spikes_per_super_synapse = count_weighted_inh_spikes / (len(inh_netcons) + 0.0)
    average_weighted_inh_spikes_per_super_synapse_per_second = count_weighted_inh_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)
    
    auxiliary_information['count_weighted_inh_spikes'] = count_weighted_inh_spikes
    auxiliary_information['average_weighted_inh_spikes_per_second'] = average_weighted_inh_spikes_per_second
    auxiliary_information['count_weighted_inh_spikes_per_super_synapse'] = count_weighted_inh_spikes_per_super_synapse
    auxiliary_information['average_weighted_inh_spikes_per_super_synapse_per_second'] = average_weighted_inh_spikes_per_super_synapse_per_second

    logger.info(f'average number of weighted inh spikes per second is {average_weighted_inh_spikes_per_second:.3f}, which is {average_weighted_inh_spikes_per_super_synapse_per_second:.3f} average weighted inh spikes per inh netcon per second')

    if inh_initial_neurons_weights is None:
        inh_initial_neurons_weights = [0.0]
    average_inh_initial_neuron_weight = np.mean(inh_initial_neurons_weights)
    auxiliary_information['inh_initial_neurons_weights'] = inh_initial_neurons_weights
    auxiliary_information['average_inh_initial_neuron_weight'] = average_inh_initial_neuron_weight

    logger.info(f'average inh initial neuron weight is {average_inh_initial_neuron_weight:.3f}')

    if inh_initial_neuron_connection_counts is not None:
        logger.info(f'min, max, avg, std, med inh initial neuron connection count are {np.min(inh_initial_neuron_connection_counts):.3f}, {np.max(inh_initial_neuron_connection_counts):.3f}, {np.mean(inh_initial_neuron_connection_counts):.3f}, {np.std(inh_initial_neuron_connection_counts):.3f}, {np.median(inh_initial_neuron_connection_counts):.3f}')
        auxiliary_information['inh_initial_neuron_connection_counts'] = inh_initial_neuron_connection_counts
    
    auxiliary_information['inh_wiring_matrix'] = inh_wiring_matrix

    return auxiliary_information

def generate_spike_times_and_weights(args, syns):
    if args.input_file is not None:
        return generate_spike_times_and_weights_from_input_file(args, syns)

    simulation_duration_in_seconds = args.simulation_duration_in_seconds

    simulation_duration_in_ms = 0
    if args.add_explicit_padding_for_initialization:
        simulation_duration_in_ms = args.simulation_initialization_duration_in_ms
    simulation_duration_in_ms += simulation_duration_in_seconds * 1000

    return generate_spike_times_and_weights_for_kernel_based_weights(args, syns, simulation_duration_in_ms)

def create_neuron_model(args):
    logger.info("About to import neuron module...")
    tm = importlib.import_module(f'{args.neuron_model_folder.replace("/",".")}.get_standard_model')
    logger.info("neuron module imported fine.")

    logger.info("About to create cell...")
    if args.max_segment_length is not None:
        cell, syns = tm.create_cell(max_segment_length=args.max_segment_length)
    else:
        cell, syns = tm.create_cell()
    logger.info("cell created fine.")

    if args.count_segments_to_stimulate is not None:
        syns = syns[:args.count_segments_to_stimulate]
        logger.info(f"Chosen {args.count_segments_to_stimulate} first segments to stimulate.")

    if args.force_number_of_segments is not None:
        logger.info(f"Currently have {len(syns)} segments and force_number_of_segments is {args.force_number_of_segments}.")
        if args.force_number_of_segments % len(syns) != 0:
            raise ForceNumberOfSegmentsIsNotAMultipleOfNumberOfSegments(f"force_number_of_segments {args.force_number_of_segments} is not a multiple of number of segments {len(syns)}.")

        multiple = int(args.force_number_of_segments / len(syns))
        logger.info(f"Multiple is {multiple}.")
        if multiple == 1:
            logger.info(f"No need to multiple segments.")
        else:
            original_number_of_segments = len(syns)
            new_values = []
            for ind in range(original_number_of_segments):
                orig_row = syns.iloc[ind]
                line = pd.DataFrame(orig_row).T
                new_values.append(line)
                for i in range(multiple - 1):
                    r = orig_row.copy()
                    r.exc_netcons = h.NetCon(None, r.exc_synapses)
                    r.inh_netcons = h.NetCon(None, r.inh_synapses)
                    line = pd.DataFrame(r).T
                    new_values.append(line)
            syns = pd.concat(new_values).reset_index(drop=True)
        logger.info(f"Now have {len(syns)} segments.")

    np_segment_lengths = np.array(syns.seg_lens)
    logger.info(f'min, max, avg, std, med segment length are {np.min(np_segment_lengths):.3f}, {np.max(np_segment_lengths):.3f}, {np.mean(np_segment_lengths):.3f}, {np.std(np_segment_lengths):.3f}, {np.median(np_segment_lengths):.3f}')

    return cell, syns


input_exc_sptimes = {}
input_inh_sptimes = {}

def run_neuron_model(args, cell, syns, simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information):

    total_number_of_netcons_after_saving = 0
    total_number_of_netcons = 0

    alt_exc_ncon_to_input_spike_times = {}
    for j, exc_ncon_and_spike_times in enumerate(exc_ncon_to_input_spike_times.items()):
        exc_netcon = exc_ncon_and_spike_times[0]
        spike_times = exc_ncon_and_spike_times[1]
        weight_to_alt_ncon = {}
        used_weights = []
        orig_exc_netcon_weight = exc_netcon.weight[0]
        orig_exc_netcon_used = False
        for sptime in spike_times:
            used_weights.append(exc_weighted_spikes[j][sptime])
            rounded_weight = round(exc_weighted_spikes[j][sptime], args.weight_rounding_precision)
            if args.use_rounded_weight:
                exc_weighted_spikes[j][sptime] = rounded_weight
            if rounded_weight in weight_to_alt_ncon:
                # reuse existing netcon, with a specific weight
                new_netcon = weight_to_alt_ncon[rounded_weight]
                alt_exc_ncon_to_input_spike_times[new_netcon] = (exc_netcon, np.concatenate((alt_exc_ncon_to_input_spike_times[new_netcon][1], np.array([sptime]))))
            else:
                if not orig_exc_netcon_used:
                    new_netcon = exc_netcon
                    orig_exc_netcon_used = True
                else:
                    new_netcon = h.NetCon(None, syns.exc_synapses[j])
                # setting the weight of the new netcon
                new_netcon.weight[0] = orig_exc_netcon_weight * (rounded_weight if args.use_rounded_weight else exc_weighted_spikes[j][sptime])
                weight_to_alt_ncon[rounded_weight] = new_netcon
                alt_exc_ncon_to_input_spike_times[new_netcon] = (exc_netcon, np.array([sptime]))

        total_number_of_netcons_after_saving += len(weight_to_alt_ncon.keys())
        total_number_of_netcons += len(used_weights)

    alt_inh_ncon_to_input_spike_times = {}
    for j, inh_ncon_and_spike_times in enumerate(inh_ncon_to_input_spike_times.items()):
        inh_netcon = inh_ncon_and_spike_times[0]
        spike_times = inh_ncon_and_spike_times[1]
        weight_to_alt_ncon = {}
        used_weights = []
        orig_inh_netcon_weight = inh_netcon.weight[0]
        orig_inh_netcon_used = False
        for sptime in spike_times:
            used_weights.append(inh_weighted_spikes[j][sptime])
            rounded_weight = round(inh_weighted_spikes[j][sptime], args.weight_rounding_precision)
            if args.use_rounded_weight:
                inh_weighted_spikes[j][sptime] = rounded_weight
            if rounded_weight in weight_to_alt_ncon:
                # reuse existing netcon, with a specific weight
                new_netcon = weight_to_alt_ncon[rounded_weight]
                alt_inh_ncon_to_input_spike_times[new_netcon] = (inh_netcon, np.concatenate((alt_inh_ncon_to_input_spike_times[new_netcon][1], np.array([sptime]))))
            else:
                if not orig_inh_netcon_used:
                    new_netcon = inh_netcon
                    orig_inh_netcon_used = True
                else:
                    new_netcon = h.NetCon(None, syns.inh_synapses[j])
                # setting the weight of the new netcon
                new_netcon.weight[0] = orig_inh_netcon_weight * (rounded_weight if args.use_rounded_weight else inh_weighted_spikes[j][sptime])
                weight_to_alt_ncon[rounded_weight] = new_netcon
                alt_inh_ncon_to_input_spike_times[new_netcon] = (inh_netcon, np.array([sptime]))

        total_number_of_netcons_after_saving += len(weight_to_alt_ncon.keys())
        total_number_of_netcons += len(used_weights)

    logger.info(f"There are {total_number_of_netcons_after_saving} netcons after saving {total_number_of_netcons-total_number_of_netcons_after_saving} out of {total_number_of_netcons}, using {args.weight_rounding_precision} precision")

    global input_exc_sptimes, input_inh_sptimes
    input_exc_sptimes = {}
    input_inh_sptimes = {}

    def apply_input_spike_times():
        logger.info("About to apply input spike times...")
        global input_exc_sptimes, input_inh_sptimes
        count_exc_events = 0
        count_inh_events = 0

        for alt_exc_netcon, exc_ncon_and_spike_times in alt_exc_ncon_to_input_spike_times.items():
            exc_netcon = exc_ncon_and_spike_times[0]
            spike_times = exc_ncon_and_spike_times[1]
            for sptime in spike_times:
                alt_exc_netcon.event(sptime)
                count_exc_events += 1
            input_exc_sptimes[exc_netcon] = spike_times

        for alt_inh_netcon, inh_ncon_and_spike_times in alt_inh_ncon_to_input_spike_times.items():
            inh_netcon = inh_ncon_and_spike_times[0]
            spike_times = inh_ncon_and_spike_times[1]
            for sptime in spike_times:
                alt_inh_netcon.event(sptime)
                count_inh_events += 1
            input_inh_sptimes[inh_netcon] = spike_times

        for exc_ncon, spike_times in exc_ncon_to_input_spike_times.items():
            input_exc_sptimes[exc_ncon] = spike_times
        for inh_ncon, spike_times in inh_ncon_to_input_spike_times.items():
            input_inh_sptimes[inh_ncon] = spike_times

        logger.info(f"Input spike applied fine, there were {count_exc_events} exc spikes and {count_inh_events} inh spikes.")

    # run sim
    cvode = h.CVode()
    if args.use_cvode:
        cvode.active(1)
    else:
        h.dt = args.dt
    h.tstop = simulation_duration_in_ms
    h.v_init = args.v_init
    fih = h.FInitializeHandler(apply_input_spike_times)
    somatic_voltage_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    time_vec = h.Vector().record(h._ref_t)

    if args.record_dendritic_voltages:
        dendritic_voltage_vecs = []
        for segment in syns.segments:
            dendritic_voltage_vec = h.Vector()
            dendritic_voltage_vec.record(segment._ref_v)
            dendritic_voltage_vecs.append(dendritic_voltage_vec)

    if args.record_synaptic_traces:
        exc_i_AMPA_vecs = []
        exc_i_NMDA_vecs = []
        exc_g_AMPA_vecs = []
        exc_g_NMDA_vecs = []
        
        inh_i_GABAA_vecs = []
        inh_i_GABAB_vecs = []
        inh_g_GABAA_vecs = []
        inh_g_GABAB_vecs = []
        
        for exc_synapse in syns.exc_synapses:
            exc_i_AMPA_vec = h.Vector()
            exc_i_AMPA_vec.record(exc_synapse._ref_i_AMPA)
            exc_i_AMPA_vecs.append(exc_i_AMPA_vec)

            exc_i_NMDA_vec = h.Vector()
            exc_i_NMDA_vec.record(exc_synapse._ref_i_NMDA)
            exc_i_NMDA_vecs.append(exc_i_NMDA_vec)

            exc_g_AMPA_vec = h.Vector()
            exc_g_AMPA_vec.record(exc_synapse._ref_g_AMPA)
            exc_g_AMPA_vecs.append(exc_g_AMPA_vec)

            exc_g_NMDA_vec = h.Vector()
            exc_g_NMDA_vec.record(exc_synapse._ref_g_NMDA)
            exc_g_NMDA_vecs.append(exc_g_NMDA_vec)

        for inh_synapse in syns.inh_synapses:
            inh_i_GABAA_vec = h.Vector()
            inh_i_GABAA_vec.record(inh_synapse._ref_i_GABAA)
            inh_i_GABAA_vecs.append(inh_i_GABAA_vec)

            inh_i_GABAB_vec = h.Vector()
            inh_i_GABAB_vec.record(inh_synapse._ref_i_GABAB)
            inh_i_GABAB_vecs.append(inh_i_GABAB_vec)

            inh_g_GABAA_vec = h.Vector()
            inh_g_GABAA_vec.record(inh_synapse._ref_g_GABAA)
            inh_g_GABAA_vecs.append(inh_g_GABAA_vec)

            inh_g_GABAB_vec = h.Vector()
            inh_g_GABAB_vec.record(inh_synapse._ref_g_GABAB)
            inh_g_GABAB_vecs.append(inh_g_GABAB_vec)         

    logger.info("Going to h.run()...")
    h_run_start_time = time.time()
    h.run()
    h_run_duration_in_seconds = time.time() - h_run_start_time
    logger.info(f"h.run() finished!, it took {h_run_duration_in_seconds/60.0:.3f} minutes")

    np_somatic_voltage_vec = np.array(somatic_voltage_vec)
    np_time_vec = np.array(time_vec)

    recording_time_low_res = np.arange(0, simulation_duration_in_ms)
    somatic_voltage_low_res = np.interp(recording_time_low_res, np_time_vec, np_somatic_voltage_vec)

    recording_time_high_res = np.arange(0, simulation_duration_in_ms, 1.0/args.count_samples_for_high_res)
    somatic_voltage_high_res = np.interp(recording_time_high_res, np_time_vec, np_somatic_voltage_vec)

    if args.record_dendritic_voltages:
        dendritic_voltages_low_res = np.zeros((len(dendritic_voltage_vecs), recording_time_low_res.shape[0]))
        dendritic_voltages_high_res = np.zeros((len(dendritic_voltage_vecs), recording_time_high_res.shape[0]))
        for segment_index, dendritic_voltage_vec in enumerate(dendritic_voltage_vecs):
            dendritic_voltages_low_res[segment_index,:] = np.interp(recording_time_low_res, np_time_vec, np.array(dendritic_voltage_vec.as_numpy()))
            dendritic_voltages_high_res[segment_index,:] = np.interp(recording_time_high_res, np_time_vec, np.array(dendritic_voltage_vec.as_numpy()))
    else:
        dendritic_voltages_low_res = None
        dendritic_voltages_high_res = None

    if args.record_synaptic_traces:
        exc_i_AMPA_low_res = np.zeros((len(exc_i_AMPA_vecs), recording_time_low_res.shape[0]))
        exc_i_NMDA_low_res = np.zeros((len(exc_i_NMDA_vecs), recording_time_low_res.shape[0]))
        exc_g_AMPA_low_res = np.zeros((len(exc_g_AMPA_vecs), recording_time_low_res.shape[0]))
        exc_g_NMDA_low_res = np.zeros((len(exc_g_NMDA_vecs), recording_time_low_res.shape[0]))

        exc_i_AMPA_high_res = np.zeros((len(exc_i_AMPA_vecs), recording_time_high_res.shape[0]))
        exc_i_NMDA_high_res = np.zeros((len(exc_i_NMDA_vecs), recording_time_high_res.shape[0]))
        exc_g_AMPA_high_res = np.zeros((len(exc_g_AMPA_vecs), recording_time_high_res.shape[0]))
        exc_g_NMDA_high_res = np.zeros((len(exc_g_NMDA_vecs), recording_time_high_res.shape[0]))

        inh_i_GABAA_low_res = np.zeros((len(inh_i_GABAA_vecs), recording_time_low_res.shape[0]))
        inh_i_GABAB_low_res = np.zeros((len(inh_i_GABAB_vecs), recording_time_low_res.shape[0]))
        inh_g_GABAA_low_res = np.zeros((len(inh_g_GABAA_vecs), recording_time_low_res.shape[0]))
        inh_g_GABAB_low_res = np.zeros((len(inh_g_GABAB_vecs), recording_time_low_res.shape[0]))

        inh_i_GABAA_high_res = np.zeros((len(inh_i_GABAA_vecs), recording_time_high_res.shape[0]))
        inh_i_GABAB_high_res = np.zeros((len(inh_i_GABAB_vecs), recording_time_high_res.shape[0]))
        inh_g_GABAA_high_res = np.zeros((len(inh_g_GABAA_vecs), recording_time_high_res.shape[0]))
        inh_g_GABAB_high_res = np.zeros((len(inh_g_GABAB_vecs), recording_time_high_res.shape[0]))

        for i, exc_i_AMPA_vec in enumerate(exc_i_AMPA_vecs):
            exc_i_AMPA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_i_AMPA_vec.as_numpy()))
            exc_i_AMPA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_i_AMPA_vec.as_numpy()))

        for i, exc_i_NMDA_vec in enumerate(exc_i_NMDA_vecs):
            exc_i_NMDA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_i_NMDA_vec.as_numpy()))
            exc_i_NMDA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_i_NMDA_vec.as_numpy()))

        for i, exc_g_AMPA_vec in enumerate(exc_g_AMPA_vecs):
            exc_g_AMPA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_g_AMPA_vec.as_numpy()))
            exc_g_AMPA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_g_AMPA_vec.as_numpy()))

        for i, exc_g_NMDA_vec in enumerate(exc_g_NMDA_vecs):
            exc_g_NMDA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_g_NMDA_vec.as_numpy()))
            exc_g_NMDA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_g_NMDA_vec.as_numpy()))

        for i, inh_i_GABAA_vec in enumerate(inh_i_GABAA_vecs):
            inh_i_GABAA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_i_GABAA_vec.as_numpy()))
            inh_i_GABAA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_i_GABAA_vec.as_numpy()))

        for i, inh_i_GABAB_vec in enumerate(inh_i_GABAB_vecs):
            inh_i_GABAB_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_i_GABAB_vec.as_numpy()))
            inh_i_GABAB_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_i_GABAB_vec.as_numpy()))

        for i, inh_g_GABAA_vec in enumerate(inh_g_GABAA_vecs):
            inh_g_GABAA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_g_GABAA_vec.as_numpy()))
            inh_g_GABAA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_g_GABAA_vec.as_numpy()))

        for i, inh_g_GABAB_vec in enumerate(inh_g_GABAB_vecs):
            inh_g_GABAB_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_g_GABAB_vec.as_numpy()))
            inh_g_GABAB_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_g_GABAB_vec.as_numpy()))

    else:
        exc_i_AMPA_low_res = None
        exc_i_NMDA_low_res = None
        exc_g_AMPA_low_res = None
        exc_g_NMDA_low_res = None
        exc_i_AMPA_high_res = None
        exc_i_NMDA_high_res = None
        exc_g_AMPA_high_res = None
        exc_g_NMDA_high_res = None
        inh_i_GABAA_low_res = None
        inh_i_GABAB_low_res = None
        inh_g_GABAA_low_res = None
        inh_g_GABAB_low_res = None
        inh_i_GABAA_high_res = None
        inh_i_GABAB_high_res = None
        inh_g_GABAA_high_res = None
        inh_g_GABAB_high_res = None    

    recordings = {}
    recordings['recording_time_low_res'] = recording_time_low_res
    recordings['somatic_voltage_low_res'] = somatic_voltage_low_res
    recordings['recording_time_high_res'] = recording_time_high_res
    recordings['somatic_voltage_high_res'] = somatic_voltage_high_res
    recordings['dendritic_voltages_low_res'] = dendritic_voltages_low_res
    recordings['dendritic_voltages_high_res'] = dendritic_voltages_high_res
    recordings['exc_i_AMPA_low_res'] = exc_i_AMPA_low_res
    recordings['exc_i_NMDA_low_res'] = exc_i_NMDA_low_res
    recordings['exc_g_AMPA_low_res'] = exc_g_AMPA_low_res
    recordings['exc_g_NMDA_low_res'] = exc_g_NMDA_low_res
    recordings['exc_i_AMPA_high_res'] = exc_i_AMPA_high_res
    recordings['exc_i_NMDA_high_res'] = exc_i_NMDA_high_res
    recordings['exc_g_AMPA_high_res'] = exc_g_AMPA_high_res
    recordings['exc_g_NMDA_high_res'] = exc_g_NMDA_high_res
    recordings['inh_i_GABAA_low_res'] = inh_i_GABAA_low_res
    recordings['inh_i_GABAB_low_res'] = inh_i_GABAB_low_res
    recordings['inh_g_GABAA_low_res'] = inh_g_GABAA_low_res
    recordings['inh_g_GABAB_low_res'] = inh_g_GABAB_low_res
    recordings['inh_i_GABAA_high_res'] = inh_i_GABAA_high_res
    recordings['inh_i_GABAB_high_res'] = inh_i_GABAB_high_res
    recordings['inh_g_GABAA_high_res'] = inh_g_GABAA_high_res
    recordings['inh_g_GABAB_high_res'] = inh_g_GABAB_high_res

    output_spike_indexes = peakutils.indexes(somatic_voltage_high_res, thres=args.spike_threshold_for_computation, thres_abs=True)
    output_spike_times = recording_time_high_res[output_spike_indexes].astype(int)

    output_data = {}

    if args.record_dendritic_voltages:
        output_data['len_dendritic_voltage_vecs'] = len(dendritic_voltage_vecs)
    if args.record_synaptic_traces:
        output_data['len_exc_i_AMPA_vecs'] = len(exc_i_AMPA_vecs)
        output_data['len_exc_i_NMDA_vecs'] = len(exc_i_NMDA_vecs)
        output_data['len_exc_g_AMPA_vecs'] = len(exc_g_AMPA_vecs)
        output_data['len_exc_g_NMDA_vecs'] = len(exc_g_NMDA_vecs)
        output_data['len_inh_i_GABAA_vecs'] = len(inh_i_GABAA_vecs)
        output_data['len_inh_i_GABAB_vecs'] = len(inh_i_GABAB_vecs)
        output_data['len_inh_g_GABAA_vecs'] = len(inh_g_GABAA_vecs)
        output_data['len_inh_g_GABAB_vecs'] = len(inh_g_GABAB_vecs)

    return output_spike_times, somatic_voltage_low_res, recordings, output_data
    
def run_actual_simulation(args, create_model_function=create_neuron_model, run_model_function=run_neuron_model):
    cell, syns = create_model_function(args)

    simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information = generate_spike_times_and_weights(args, syns)

    output_spike_times, somatic_voltage_low_res, recordings, output_data = run_model_function(args, cell, syns, simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information)

    # relevant when using a non NEURON model
    recordings['somatic_voltage_low_res'] = somatic_voltage_low_res

    output_firing_rate = len(output_spike_times)/(simulation_duration_in_ms/1000.0)
    output_isi = np.diff(output_spike_times)
    
    output_spike_times_after_initialization = output_spike_times[output_spike_times > args.simulation_initialization_duration_in_ms]
    output_firing_rate_after_initialization = len(output_spike_times_after_initialization)/((simulation_duration_in_ms - args.simulation_initialization_duration_in_ms)/1000.0)
    output_isi_after_initialization = np.diff(output_spike_times)

    average_somatic_voltage = np.mean(somatic_voltage_low_res)

    clipped_somatic_voltage_low_res = np.copy(somatic_voltage_low_res)
    clipped_somatic_voltage_low_res[clipped_somatic_voltage_low_res>args.spike_threshold] = args.spike_threshold
    average_clipped_somatic_voltage = np.mean(clipped_somatic_voltage_low_res)

    output_data['args'] = args

    output_data['len_exc_netcons'] = len(syns.exc_netcons)
    output_data['len_inh_netcons'] = len(syns.inh_netcons)

    output_data['input_count_exc_spikes'] = auxiliary_information['count_exc_spikes']
    output_data['input_average_exc_spikes_per_second'] = auxiliary_information['average_exc_spikes_per_second']
    output_data['input_count_exc_spikes_per_super_synapse'] = auxiliary_information['count_exc_spikes_per_super_synapse']
    output_data['input_average_exc_spikes_per_super_synapse_per_second'] = auxiliary_information['average_exc_spikes_per_super_synapse_per_second']
    output_data['input_count_weighted_exc_spikes'] = auxiliary_information['count_weighted_exc_spikes']
    output_data['input_average_weighted_exc_spikes_per_second'] = auxiliary_information['average_weighted_exc_spikes_per_second']
    output_data['input_count_weighted_exc_spikes_per_super_synapse'] = auxiliary_information['count_weighted_exc_spikes_per_super_synapse']
    output_data['input_average_weighted_exc_spikes_per_super_synapse_per_second'] = auxiliary_information['average_weighted_exc_spikes_per_super_synapse_per_second']
    output_data['input_count_inh_spikes'] = auxiliary_information['count_inh_spikes']
    output_data['input_average_inh_spikes_per_second'] = auxiliary_information['average_inh_spikes_per_second']
    output_data['input_count_inh_spikes_per_super_synapse'] = auxiliary_information['count_inh_spikes_per_super_synapse']
    output_data['input_average_inh_spikes_per_super_synapse_per_second'] = auxiliary_information['average_inh_spikes_per_super_synapse_per_second']
    output_data['input_count_weighted_inh_spikes'] = auxiliary_information['count_weighted_inh_spikes']
    output_data['input_average_weighted_inh_spikes_per_second'] = auxiliary_information['average_weighted_inh_spikes_per_second']
    output_data['input_count_weighted_inh_spikes_per_super_synapse'] = auxiliary_information['count_weighted_inh_spikes_per_super_synapse']
    output_data['input_average_weighted_inh_spikes_per_super_synapse_per_second'] = auxiliary_information['average_weighted_inh_spikes_per_super_synapse_per_second']

    output_data['average_exc_initial_neuron_weight'] = auxiliary_information['average_exc_initial_neuron_weight']
    output_data['average_inh_initial_neuron_weight'] = auxiliary_information['average_inh_initial_neuron_weight']
    
    output_data['output_spike_times'] = output_spike_times

    output_data['output_firing_rate'] = output_firing_rate
    output_data['output_isi'] = output_isi
    output_data['output_spike_times_after_initialization'] = output_spike_times_after_initialization
    output_data['output_firing_rate_after_initialization'] = output_firing_rate_after_initialization
    output_data['output_isi_after_initialization'] = output_isi_after_initialization
    
    output_data['simulation_duration_in_ms'] = simulation_duration_in_ms
    output_data['average_somatic_voltage'] = average_somatic_voltage
    output_data['average_clipped_somatic_voltage'] = average_clipped_somatic_voltage
    
    if args.save_auxiliary_information:
        output_data['auxiliary_information'] = auxiliary_information

    return output_data, exc_weighted_spikes, inh_weighted_spikes, recordings, auxiliary_information, cell, syns

def run_simulation(args, create_model_function=create_neuron_model, run_model_function=run_neuron_model):
    logger.info("Going to run simulation with args:")
    logger.info("{}".format(args))
    logger.info("...")

    logger.info("After shortcuts, args are:")
    logger.info("{}".format(args))

    os.makedirs(args.simulation_folder, exist_ok=True)

    run_simulation_start_time = time.time()

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)

    if args.neuron_model_folder is not None:
        # trying to fix neuron crashes
        time.sleep(1 + 30*np.random.random())

        logger.info("About to import neuron...")
        logger.info(f"current dir: {pathlib.Path(__file__).parent.absolute()}")
        
        global neuron
        global h
        global gui
        import neuron
        from neuron import h
        from neuron import gui
        logger.info("Neuron imported fine.")

    simulation_trial = 0
    output_data, exc_weighted_spikes, inh_weighted_spikes, recordings, auxiliary_information, cell, syns = run_actual_simulation(args, create_model_function=create_model_function, run_model_function=run_model_function)
    output_firing_rate = output_data['output_firing_rate']
    output_firing_rate_after_initialization = output_data['output_firing_rate_after_initialization']
    simulation_trial += 1

    while output_firing_rate <= 0.0 and simulation_trial < args.count_trials_for_nonzero_output_firing_rate:
        logger.info(f"Firing rate is {output_firing_rate:.3f}, Firing rate after initialization is {output_firing_rate_after_initialization:.3f}")
        logger.info(f"Retrying simulation, {simulation_trial} trial")
        output_data, exc_weighted_spikes, inh_weighted_spikes, recordings, auxiliary_information, cell, syns = run_actual_simulation(args, create_model_function=create_model_function, run_model_function=run_model_function)
        output_firing_rate = output_data['output_firing_rate']
        output_firing_rate_after_initialization = output_data['output_firing_rate_after_initialization']
        simulation_trial += 1

    logger.info(f"Firing rate is {output_firing_rate:.3f}, Firing rate after initialization is {output_firing_rate_after_initialization:.3f}")
    logger.info(f"output_spike_times are {output_data['output_spike_times']}")
    logger.info(f"Simulation finished after {simulation_trial} trials")

    pickle.dump(output_data, open(f'{args.simulation_folder}/summary.pkl','wb'), protocol=-1)

    sparse.save_npz(f'{args.simulation_folder}/exc_weighted_spikes.npz', sparse.csr_matrix(exc_weighted_spikes))
    sparse.save_npz(f'{args.simulation_folder}/inh_weighted_spikes.npz', sparse.csr_matrix(inh_weighted_spikes))

    f = h5py.File(f'{args.simulation_folder}/voltage.h5','w')
    f.create_dataset('somatic_voltage', data=recordings['somatic_voltage_low_res'])
    if args.record_dendritic_voltages:
        f.create_dataset('dendritic_voltage', data=recordings['dendritic_voltages_low_res'])
    if args.record_synaptic_traces:
        f.create_dataset('exc_i_AMPA', data=recordings['exc_i_AMPA_low_res'])
        f.create_dataset('exc_i_NMDA', data=recordings['exc_i_NMDA_low_res'])
        f.create_dataset('exc_g_AMPA', data=recordings['exc_g_AMPA_low_res'])
        f.create_dataset('exc_g_NMDA', data=recordings['exc_g_NMDA_low_res'])
        f.create_dataset('inh_i_GABAA', data=recordings['inh_i_GABAA_low_res'])
        f.create_dataset('inh_i_GABAB', data=recordings['inh_i_GABAB_low_res'])
        f.create_dataset('inh_g_GABAA', data=recordings['inh_g_GABAA_low_res'])
        f.create_dataset('inh_g_GABAB', data=recordings['inh_g_GABAB_low_res'])
    f.close()

    if args.save_plots:
        # io plot
        ws = np.vstack((exc_weighted_spikes, inh_weighted_spikes))
        half_syn = exc_weighted_spikes.shape[0]
        count_spikes = len(output_data['output_spike_times'])
        name = os.path.basename(args.simulation_folder)
        avg_exc = output_data['input_average_weighted_exc_spikes_per_super_synapse_per_second']
        avg_inh = output_data['input_average_weighted_inh_spikes_per_super_synapse_per_second']

        if 'recording_time_high_res' in recordings and 'somatic_voltage_high_res' in recordings:
            recording_time_high_res = recordings['recording_time_high_res']
            somatic_voltage_high_res = recordings['somatic_voltage_high_res']
        else:
            # for non NEURON models
            somatic_voltage_high_res = recordings['somatic_voltage_low_res']
            recording_time_high_res = np.array(range(somatic_voltage_high_res.shape[0]))

        max_weight = ws.max()

        is_weighted = False

        plot_input_spikes = is_weighted and 'exc_initial_neurons_spikes_bin' in auxiliary_information and 'inh_initial_neurons_spikes_bin' in auxiliary_information

        if plot_input_spikes:
            fig = plt.figure(figsize=(25,15))
            axs = fig.subplots(3,1, sharex=True)
        else:
            fig = plt.figure(figsize=(25,10))
            axs = fig.subplots(2,1, sharex=True)

        fig.suptitle(f'{name}\nAverage input per segment: {avg_exc:.3f} exc Hz, {avg_inh:.3f} inh Hz\n'+
        f'Output: {count_spikes} spikes ({output_firing_rate:.3f} Hz)', fontsize=20)

        if plot_input_spikes:
            input_spikes = np.vstack((auxiliary_information['exc_initial_neurons_spikes_bin'], auxiliary_information['inh_initial_neurons_spikes_bin']))
            half_axon = auxiliary_information['exc_initial_neurons_spikes_bin'].shape[0]
            axs[0].matshow(input_spikes, cmap='binary', aspect='auto')
            axs[0].xaxis.set_ticks_position("bottom")
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Axon')
            axs[0].axhline(half_axon, color='green')
            axs[0].set_title('Input Spikes')

        if max_weight > 1:
            first = int(128 / (max_weight-1))
            colors1 = plt.cm.binary(np.linspace(0., 1, first))
            colors2 = plt.cm.hot(np.linspace(0, 0.8, 256-first))
            colors = np.vstack((colors1, colors2))
            mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            chosen_cmap = mymap
        else:
            chosen_cmap = plt.cm.binary

        # divider = make_axes_locatable(axs[0])
        # cax = divider.append_axes('right', size='5%', pad=0.05)

        # axs[0].matshow(ws, cmap='Reds', vmin=0, vmax=5)
        # axs[0].matshow(ws, cmap='binary', vmin=0, vmax=1)

        ax_index = 1 if plot_input_spikes else 0

        im = axs[ax_index].matshow(ws, cmap=chosen_cmap, vmin=0, vmax=max_weight, aspect='auto')
        if is_weighted:
            cax = fig.add_axes([0.27, 0.80, 0.5, 0.05])
            fig.colorbar(im, cax=cax, orientation='horizontal')
        # fig.colorbar(im, cax=cax, orientation='vertical')

        axs[ax_index].axhline(half_syn, color='green')
        axs[ax_index].xaxis.set_ticks_position("bottom")
        axs[ax_index].set_xlabel('Time (ms)')
        axs[ax_index].set_ylabel('Segment')

        if is_weighted:
            axs[ax_index].set_title('Input Weighted spikes')
        else:
            axs[ax_index].set_title('Input Spikes')

        ax_index += 1
        axs[ax_index].plot(recording_time_high_res, somatic_voltage_high_res)
        axs[ax_index].set_xlabel('Time (ms)')
        axs[ax_index].set_ylabel('Voltage (mV)')
        # axs[ax_index].set_xlim(0, 3000)
        axs[ax_index].set_xlim(0, exc_weighted_spikes.shape[1])
        axs[ax_index].set_title('Output Somatic voltage')
        if is_weighted:
            plt.subplots_adjust(bottom=0.2, top=0.73, hspace=0.35)
        else:
            plt.subplots_adjust(bottom=0.2, top=0.82, hspace=0.3)
        fig.savefig(f'{args.simulation_folder}/io.png')
        plt.close('all')

        exc_wiring_matrix = auxiliary_information['exc_wiring_matrix']
        inh_wiring_matrix = auxiliary_information['inh_wiring_matrix']
        if exc_wiring_matrix is not None and inh_wiring_matrix is not None:
            fig = plt.figure(figsize=(10,10))
            axs = fig.subplots(1,2)
            axs[0].matshow(exc_wiring_matrix, cmap='hot', aspect='auto')
            exc_stats_string = f"avg per row is {np.mean(np.sum(exc_wiring_matrix, axis=1)):.3f}, avg per col is {np.mean(np.sum(exc_wiring_matrix, axis=0)):.3f}"
            axs[0].set_title(f'Exc wiring matrix\n{exc_stats_string}')
            axs[0].xaxis.set_ticks_position('bottom')
            axs[0].set_xlabel("axon")
            axs[0].set_ylabel("segment")

            axs[1].matshow(inh_wiring_matrix, cmap='hot', aspect='auto')
            inh_stats_string = f"avg per row is {np.mean(np.sum(inh_wiring_matrix, axis=1)):.3f}, avg per col is {np.mean(np.sum(inh_wiring_matrix, axis=0)):.3f}"
            axs[1].set_title(f'Inh wiring matrix\n{inh_stats_string}')
            axs[1].xaxis.set_ticks_position('bottom')
            axs[1].set_xlabel("axon")
            axs[1].set_ylabel("segment")
            fig.savefig(f'{args.simulation_folder}/wiring.png')
            plt.close('all')

        
    run_simulation_duration_in_seconds = time.time() - run_simulation_start_time
    logger.info(f"run simulation finished!, it took {run_simulation_duration_in_seconds/60.0:.3f} minutes")

    if args.finish_file:
        with open(args.finish_file, 'w') as f:
            f.write('finished')
    
    return run_simulation_duration_in_seconds

def get_simulation_args():
    saver = ArgumentSaver()
    saver.add_argument('--simulation_duration_in_seconds', default=10, type=int)
    saver.add_argument('--random_seed', default=None, type=int)

    saver.add_argument('--max_segment_length', default=None, type=float)
    saver.add_argument('--count_segments_to_stimulate', default=None, type=int)
    saver.add_argument('--force_number_of_segments', default=None, type=int)

    saver.add_argument('--use_cvode', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--simulation_initialization_duration_in_ms', default=500, type=int)
    saver.add_argument('--zero_padding_for_initialization', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--add_explicit_padding_for_initialization', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--count_samples_for_high_res', default=8, type=int)
    saver.add_argument('--record_dendritic_voltages', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--record_synaptic_traces', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--save_auxiliary_information', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--dt', default=0.025, type=float)
    saver.add_argument('--v_init', default=-76.0, type=float)
    saver.add_argument('--spike_threshold_for_computation', default=-20, type=float)
    saver.add_argument('--spike_threshold', default=-55, type=float)

    saver.add_argument('--use_rounded_weight', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--weight_rounding_precision', default=5, type=int)

    # number of spike ranges for the simulation
    saver.add_argument('--count_exc_spikes_per_synapse_per_100ms_range', nargs='+', type=float, default=[0, 0.1]) # up to average 1Hz
    saver.add_argument('--count_inh_spikes_per_synapse_per_100ms_range', nargs='+', type=float, default=[0, 0.1]) # up to average 1Hz
    saver.add_argument('--adaptive_inh', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--adaptive_inh_additive_factor_per_synapse_per_100ms_range', nargs='+', type=float, default=[-0.07, 0.03])
    
    saver.add_argument('--count_trials_for_nonzero_output_firing_rate', default=1, type=int)
    saver.add_argument('--force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length', type=str2bool, nargs='?', const=True, default=False)
    
    # define inst rate between change interval and smoothing sigma options (two rules of thumb:)
    # (A) increasing sampling time interval increases firing rate (more cumulative spikes at "lucky high rate" periods)
    # (B) increasing smoothing sigma reduces output firing rate (reduce effect of "lucky high rate" periods due to averaging)
    saver.add_argument('--inst_rate_sampling_time_interval_options_ms', nargs='+', type=int, default=[25,30,35,40,45,50,55,60,65,70,75,80,85,90,100,150,200,300,450])
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_options_ms', nargs='+', type=int, default=[25,30,35,40,45,50,55,60,65,80,100,150,200,250,300,400,500,600])
    saver.add_argument('--inst_rate_sampling_time_interval_jitter_range', default=20, type=int)
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_jitter_range', default=20, type=int)
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_mult', default=7.0, type=float)

    saver.add_argument('--spatial_multiplicative_randomness', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--exc_spatial_multiplicative_randomness_delta_prob', default=0.85, type=float)
    saver.add_argument('--inh_spatial_multiplicative_randomness_delta_prob', default=0.85, type=float)
    saver.add_argument('--exc_spatial_multiplicative_randomness_delta_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--inh_spatial_multiplicative_randomness_delta_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--same_exc_inh_spatial_multiplicative_randomness_delta_prob', default=0.7, type=float)

    # weight generation parameters
    saver.add_argument('--exc_weights_ratio_range', nargs='+', type=float, default=[1.0, 1.0])
    saver.add_argument('--inh_weights_ratio_range', nargs='+', type=float, default=[1.0, 1.0])

    # count of initial synapses per super synapse parameters
    saver.add_argument('--force_count_initial_synapses_per_super_synapse', default=None, type=int)
    saver.add_argument('--force_count_initial_synapses_per_tree', default=None, type=int)

    return saver

def get_args():
    parser = argparse.ArgumentParser(description='Simulate a neuron')
    parser.add_argument('--neuron_model_folder')
    parser.add_argument('--simulation_folder', action=AddOutFileAction)
    parser.add_argument('--input_file', default=None)
    
    saver = get_simulation_args()
    saver.add_to_parser(parser)
    
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)
    return parser.parse_args()

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())

    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to neuron simulator! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    run_simulation(args)
    logger.info(f"Goodbye from neuron simulator! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
