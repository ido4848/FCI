from __future__ import print_function
import pathlib
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import argparse
import heapq

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.status import Status
from utils.slurm_job import SlurmJobFactory, get_job_submitter_args, get_job_args
from utils.utils import setup_logger, str2bool, args_to_arg_string, MAXIMAL_RANDOM_SEED, ArgumentSaver, AddDefaultInformationAction, AddOutFileAction, TeeAll
from simulating_neurons.simulate_neuron import get_simulation_args

import logging
logger = logging.getLogger(__name__)

NEURON_FILE_TO_RUN = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "simulate_neuron.py")

def get_generate_dataset_args():
    saver = ArgumentSaver()
    
    saver.add_argument('--train_input_folder', default=None)
    saver.add_argument('--valid_input_folder', default=None)
    saver.add_argument('--test_input_folder', default=None)

    saver.add_argument('--count_simulations_for_train', default=10000, type=int, action=AddDefaultInformationAction) # defaults for fci
    saver.add_argument('--count_simulations_for_valid', default=10, type=int, action=AddDefaultInformationAction)
    saver.add_argument('--count_simulations_for_test', default=2500, type=int, action=AddDefaultInformationAction) # default for fci

    saver.add_argument('--percent_simulations_for_normalized_valid', default=0.4, type=float)
    saver.add_argument('--normalized_valid_average_firing_rate', default=1.0, type=float)
    saver.add_argument('--normalized_valid_average_firing_rate_margin', default=0.001, type=float)
    saver.add_argument('--normalized_valid_normalize_isi', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--normalized_valid_average_isi', default=550, type=float)
    saver.add_argument('--normalized_valid_average_isi_margin', default=1, type=float)
    saver.add_argument('--normalized_valid_choosing_change_per_iteration_fr', default=50, type=int)
    saver.add_argument('--normalized_valid_choosing_change_per_iteration_isi', default=1, type=int)
    saver.add_argument('--normalized_valid_choosing_maximum_strikes', default=500, type=int)

    saver.add_argument('--percent_simulations_for_normalized_test', default=0.4, type=float)
    saver.add_argument('--normalized_test_average_firing_rate', default=1.0, type=float)
    saver.add_argument('--normalized_test_average_firing_rate_margin', default=0.001, type=float)
    saver.add_argument('--normalized_test_normalize_isi', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--normalized_test_average_isi', default=550, type=float)
    saver.add_argument('--normalized_test_average_isi_margin', default=1, type=float)
    saver.add_argument('--normalized_test_choosing_change_per_iteration_fr', default=50, type=int)
    saver.add_argument('--normalized_test_choosing_change_per_iteration_isi', default=1, type=int)
    saver.add_argument('--normalized_test_choosing_maximum_strikes', default=500, type=int)

    # number of spikes generation parameters
    saver.add_argument('--exc_fr_per_synapse_options', nargs='+', type=float, default=[0.0, 1.0, 1.0, 2.0])
    saver.add_argument('--inh_fr_per_synapse_options', nargs='+', type=float, default=[0.0, 1.0, 1.0, 2.0])
    saver.add_argument('--fr_option_choice_by_modulo', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--same_exc_inh_count_spikes_per_synapse_per_100ms_range_prob', type=float, default=0.1)

    # weight generation parameters, default is unweighted
    saver.add_argument('--exc_weight_ratio_options', nargs='+', type=float, default=[1.0, 1.0])
    saver.add_argument('--inh_weight_ratio_options', nargs='+', type=float, default=[1.0, 1.0])

    saver.add_argument('--same_exc_inh_weight_ratio_range_prob', type=float, default=0.15)

    # count of initial synapses per super synapse parameters
    saver.add_argument('--count_exc_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max', nargs='+', type=float, default=[0.7, 0.5, 1.0])
    saver.add_argument('--count_exc_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min', nargs='+', type=float, default=[1.5, 2.0, 1.0])
    saver.add_argument('--count_inh_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max', nargs='+', type=float, default=[0.7, 0.5, 1.0])
    saver.add_argument('--count_inh_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min', nargs='+', type=float, default=[1.5, 2.0, 1.0])
    saver.add_argument('--same_exc_inh_initial_synapses_per_super_synapse_mult_factor_range_prob', type=float, default=0.6)

    saver.add_argument('--simulation_job_timelimit', default=60*30, type=int) # 30 minutes in seconds
    
    saver.add_argument('--no_save_plots_for_simulations', type=str2bool, nargs='?', const=True, default=True)

    saver.add_argument('--summarize_output_spike_times', type=str2bool, nargs='?', const=True, default=False)

    simulation_saver = get_simulation_args()
    simulation_saver.add_to_parser(saver)

    return saver

def get_generate_dataset_parser():
    parser = argparse.ArgumentParser(description='Generate dataset for a neuron')
    parser.add_argument('--neuron_model_folder')
    parser.add_argument('--simulation_dataset_folder', action=AddOutFileAction)
    parser.add_argument('--simulation_dataset_name')
    
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)

    saver = get_generate_dataset_args()
    saver.add_to_parser(parser)
    
    job_saver = get_job_args()
    job_saver.add_to_parser(parser)

    job_submitter_saver = get_job_submitter_args()
    job_submitter_saver.add_to_parser(parser)

    return parser
    
def get_args():
    parser = get_generate_dataset_parser()
    return parser.parse_args()

class GenerateDatasetStatus(Status):
    def __init__(self, args):
        super().__init__(args)

        total_number_of_simulations = args.count_simulations_for_train + args.count_simulations_for_valid + args.count_simulations_for_test
        all_random_seeds = list(set(np.random.randint(0, MAXIMAL_RANDOM_SEED + 1, 2 * total_number_of_simulations))) # remove duplications
        self.all_random_seeds = all_random_seeds
        self.random_seed = {"train":{}, "valid":{}, "test":{}}
        
        self.finished = {"train":{}, "valid":{}, "test":{}}

        self.simulation_duration_in_ms = 0
        self.len_exc_netcons = 0
        self.len_inh_netcons = 0
        self.average_somatic_voltage = {"train":{}, "valid":{}, "test":{}}
        self.average_clipped_somatic_voltage = {"train":{}, "valid":{}, "test":{}}
        self.output_firing_rate_after_initialization = {"train":{}, "valid":{}, "test":{}}
        self.output_isi_after_initialization = {"train":{}, "valid":{}, "test":{}}
        self.input_average_exc_spikes_per_super_synapse_per_second = {"train":{}, "valid":{}, "test":{}}
        self.input_average_inh_spikes_per_super_synapse_per_second = {"train":{}, "valid":{}, "test":{}}
        self.input_average_weighted_exc_spikes_per_super_synapse_per_second = {"train":{}, "valid":{}, "test":{}}
        self.input_average_weighted_inh_spikes_per_super_synapse_per_second = {"train":{}, "valid":{}, "test":{}}
        self.average_exc_initial_neuron_weight = {"train":{}, "valid":{}, "test":{}}
        self.average_inh_initial_neuron_weight = {"train":{}, "valid":{}, "test":{}}

        if args.summarize_output_spike_times:
            self.output_spike_times = {"train":{}, "valid":{}, "test":{}}
        
        for state in ["train", "valid", "test"]:
            count_simulations = self.kwargs["count_simulations_for_{}".format(state)]
            for i in range(count_simulations):
                self.finished[state][i] = False
                self.random_seed[state][i] = all_random_seeds.pop()

        self.save()

    def all_finished(self):
        for state in ["train", "valid", "test"]:
            count_simulations = self.kwargs["count_simulations_for_{}".format(state)]
            for i in range(count_simulations):
                if not self.finished[state][i]:
                    return False
        return True

def choose_normalized_fr_isi_set(set_by_index, kwargs, name, log_iter=True):
    logger.info(f"Going to choose normalized {name} set...")
    choosing_start_time = time.time()

    count_to_take = kwargs[f'count_simulations_for_normalized_{name}']
    fr_needed_avg = kwargs[f'normalized_{name}_average_firing_rate']
    fr_margin = kwargs[f'normalized_{name}_average_firing_rate_margin']
    normalize_isi = kwargs[f'normalized_{name}_normalize_isi']
    isi_needed_avg = kwargs[f'normalized_{name}_average_isi']
    isi_margin = kwargs[f'normalized_{name}_average_isi_margin']
    fr_iter_change = kwargs[f'normalized_{name}_choosing_change_per_iteration_fr']
    isi_iter_change = kwargs[f'normalized_{name}_choosing_change_per_iteration_isi']
    maximum_strikes =  kwargs[f'normalized_{name}_choosing_maximum_strikes']

    taken_indices = np.random.choice(list(range(len(set_by_index))), count_to_take, replace=False)
    taken = {index: set_by_index[index] for index in taken_indices}
    set_by_index_without_taken = {index: set_by_index[index] for index in list(set(range(len(set_by_index))) - set(taken_indices))}

    def get_avg_isi(taken):
        isi_values = np.concatenate(list([d['isi'] for d in taken.values()]))
        if len(isi_values) == 0:
            isi_values = np.array([0.0])
        return np.mean(isi_values)

    def get_avg_fr(taken):
        return np.mean(list([d['fr'] for d in taken.values()]))

    def get_isi_key(isi_list_list):
        def isi_key(idx):
            isi_list = isi_list_list[idx]['isi']
            if len(isi_list) == 0:
                return 0
            return np.mean(isi_list)
        return isi_key

    def get_fr_key(fr_list):
        def fr_key(idx):
            return fr_list[idx]['fr']
        return fr_key

    iter = 0
    strikes = 0
    
    logger.info(f'initial avgs are {get_avg_fr(taken):.5f}, {get_avg_isi(taken):.5f}, len(set_by_index_without_taken) is {len(set_by_index_without_taken)}')    
    while (abs(get_avg_fr(taken) - fr_needed_avg) > fr_margin or (normalize_isi and abs(get_avg_isi(taken) - isi_needed_avg) > isi_margin))\
         and iter < kwargs[f'normalized_{name}_choosing_max_iteration']:
        iter += 1
        
        initial_avg_fr = get_avg_fr(taken)
        avg_fr_minus_needed = initial_avg_fr - fr_needed_avg
        first_sign = '-' if avg_fr_minus_needed > fr_margin else '+'
        should_change_fr = abs(get_avg_fr(taken) - fr_needed_avg) > fr_margin
        cur_fr_iter_change = fr_iter_change
        if not should_change_fr:
            first_sign = '0'
            cur_fr_iter_change = len(set_by_index_without_taken)
            
        
        should_change_isi = normalize_isi and abs(get_avg_isi(taken) - isi_needed_avg) > isi_margin
        initial_avg_isi = get_avg_isi(taken)
        avg_isi_minus_needed = initial_avg_isi - isi_needed_avg
        second_sign = '-' if avg_isi_minus_needed > isi_margin else '+'
        if not should_change_isi:
            second_sign = '0'
            cur_fr_iter_change = 1


        if initial_avg_fr - fr_needed_avg > fr_margin:
            fr_indices_in_set_by_index_without_taken = heapq.nsmallest(cur_fr_iter_change, set_by_index_without_taken, key=get_fr_key(set_by_index_without_taken))
            if len(fr_indices_in_set_by_index_without_taken) == 0:
                break
            fr_indices_in_taken = heapq.nlargest(len(fr_indices_in_set_by_index_without_taken), taken, key=get_fr_key(taken))
            if len(fr_indices_in_taken) == 0:
                break

        else:
            fr_indices_in_set_by_index_without_taken = heapq.nlargest(cur_fr_iter_change, set_by_index_without_taken, key=get_fr_key(set_by_index_without_taken))
            if len(fr_indices_in_set_by_index_without_taken) == 0:
                break
            fr_indices_in_taken = heapq.nsmallest(len(fr_indices_in_set_by_index_without_taken), taken, key=get_fr_key(taken))
            if len(fr_indices_in_taken) == 0:
                break

        isi_candidates_set_by_index_without_taken = {index:set_by_index_without_taken[index] for index in fr_indices_in_set_by_index_without_taken}
        isi_candidates_taken = {index:taken[index] for index in fr_indices_in_taken}
        if avg_isi_minus_needed > isi_margin:
            isi_indices_in_set_by_index_without_taken = heapq.nsmallest(isi_iter_change, isi_candidates_set_by_index_without_taken, key=get_isi_key(isi_candidates_set_by_index_without_taken))
            if len(isi_indices_in_set_by_index_without_taken) == 0:
                break
            isi_indices_in_taken = heapq.nlargest(len(isi_indices_in_set_by_index_without_taken), isi_candidates_taken, key=get_isi_key(isi_candidates_taken))
            if len(isi_indices_in_taken) == 0:
                break
        else:
            isi_indices_in_set_by_index_without_taken = heapq.nlargest(isi_iter_change, isi_candidates_set_by_index_without_taken, key=get_isi_key(isi_candidates_set_by_index_without_taken))
            if len(isi_indices_in_set_by_index_without_taken) == 0:
                break
            isi_indices_in_taken = heapq.nsmallest(len(isi_indices_in_set_by_index_without_taken), isi_candidates_taken, key=get_isi_key(isi_candidates_taken))
            if len(isi_indices_in_taken) == 0:
                break

        for index in isi_indices_in_taken:
            del taken[index]
        for index in isi_indices_in_set_by_index_without_taken:
            taken[index] = set_by_index_without_taken[index]
            del set_by_index_without_taken[index]
          
        avg_fr = get_avg_fr(taken)
        avg_isi = get_avg_isi(taken)

        first_strike = False
        if first_sign == '-' and avg_fr > initial_avg_fr:
            first_strike = True
            strikes += 1
        if first_sign == '+' and avg_fr < initial_avg_fr:
            first_strike = True
            strikes += 1
        if first_sign == '0':
            first_strike = True

        second_strike = False
        if second_sign == '-' and avg_isi > initial_avg_isi:
            second_strike = True
            strikes += 1
        if second_sign == '+' and avg_isi < initial_avg_isi:
            second_strike = True
            strikes += 1
        if second_sign == '0':
            second_strike = True

        if first_strike and second_strike:
            logger.info(f'out first_strike & second_strike')
            break

        if strikes > maximum_strikes:
            logger.info(f'out {strikes} > {maximum_strikes}')
            break

        if log_iter:
            logger.info(f'after {first_sign}, {second_sign}, avgs are {avg_fr:.5f}, {avg_isi:.5f}, len(set_by_index_without_taken) is {len(set_by_index_without_taken)}')
            
    final_avg_isi = get_avg_isi(taken)
    final_avg_fr = get_avg_fr(taken)
    logger.info(f'final avgs are {final_avg_fr:.5f}, {final_avg_isi:.5f} after {iter} iterations, for {count_to_take} simulations')

    choosing_duration_in_seconds = time.time() - choosing_start_time
    logger.info(f"choose normalized {name} set finished!, it took {choosing_duration_in_seconds:.3f} seconds")

    return taken, final_avg_fr, final_avg_isi

def neuron_extra_arg_string_func(args):
    return f'--neuron_model_folder {args.neuron_model_folder}'

def generate_dataset(args, file_to_run=NEURON_FILE_TO_RUN, extra_arg_string_func=neuron_extra_arg_string_func):
    logger.info("Going to generate dataset with args:")
    logger.info("{}".format(args))
    logger.info("...")

    # TODO: is this assert needed?
    # assert args.count_simulations_for_test >= 10, "must have at least 10 test simulations"
    args.count_simulations_for_normalized_test = max(1, int(np.floor(args.percent_simulations_for_normalized_test * args.count_simulations_for_test)))
    args.normalized_test_choosing_max_iteration = max(1, args.count_simulations_for_test - args.count_simulations_for_normalized_test - 1) # allow some margin

    # TODO: is this assert needed?
    # assert args.count_simulations_for_valid >= 10, "must have at least 10 valid simulations"
    args.count_simulations_for_normalized_valid = max(1, int(np.floor(args.percent_simulations_for_normalized_valid * args.count_simulations_for_valid)))
    args.normalized_valid_choosing_max_iteration = max(1, args.count_simulations_for_valid - args.count_simulations_for_normalized_valid - 1) # allow some margin

    assert len(args.exc_fr_per_synapse_options) == len(args.inh_fr_per_synapse_options), "must have the same number of exc and inh fr options"
    assert len(args.exc_fr_per_synapse_options) % 2 == 0, "list of options must be even (low, high)"

    assert len(args.exc_weight_ratio_options) == len(args.inh_weight_ratio_options), "must have the same number of exc and inh weight options"
    assert len(args.exc_weight_ratio_options) % 2 == 0, "list of weight options must be even (low, high)"

    logger.info("After shortcuts, args are:")
    logger.info("{}".format(args))

    kwargs = vars(args)

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)

    args.save_folder = args.simulation_dataset_folder
    status = GenerateDatasetStatus.get_status(args)

    simulation_dataset_folder = args.simulation_dataset_folder

    simulation_dataset_basename = args.simulation_dataset_name
    if not simulation_dataset_basename:
        simulation_dataset_basename = os.path.basename(simulation_dataset_folder)
    
    status.start()

    os.makedirs(simulation_dataset_folder, exist_ok=True)

    def on_join(job_state, extra):
        if job_state.is_successfull():
            state = extra[0]
            i = extra[1]
            status.finished[state][i] = True
            
            state_folder = os.path.join(simulation_dataset_folder, state)
            simulation_folder = os.path.join(state_folder, f"simulation_{i}")
            current_summary = pickle.load(open(f"{simulation_folder}/summary.pkl", "rb"))

            if i == 0:
                status.simulation_duration_in_ms = current_summary["simulation_duration_in_ms"]
                status.len_exc_netcons = current_summary["len_exc_netcons"]
                status.len_inh_netcons = current_summary["len_inh_netcons"]

            status.average_somatic_voltage[state][i] = current_summary["average_somatic_voltage"]
            status.average_clipped_somatic_voltage[state][i] = current_summary["average_clipped_somatic_voltage"]
            status.output_firing_rate_after_initialization[state][i] = current_summary["output_firing_rate_after_initialization"]
            status.output_isi_after_initialization[state][i] = current_summary["output_isi_after_initialization"]
            status.input_average_exc_spikes_per_super_synapse_per_second[state][i] = current_summary["input_average_exc_spikes_per_super_synapse_per_second"]
            status.input_average_inh_spikes_per_super_synapse_per_second[state][i] = current_summary["input_average_inh_spikes_per_super_synapse_per_second"]
            status.input_average_weighted_exc_spikes_per_super_synapse_per_second[state][i] = current_summary["input_average_weighted_exc_spikes_per_super_synapse_per_second"]
            status.input_average_weighted_inh_spikes_per_super_synapse_per_second[state][i] = current_summary["input_average_weighted_inh_spikes_per_super_synapse_per_second"]
            status.average_exc_initial_neuron_weight[state][i] = current_summary["average_exc_initial_neuron_weight"]
            status.average_inh_initial_neuron_weight[state][i] = current_summary["average_inh_initial_neuron_weight"]

            if args.summarize_output_spike_times:
                status.output_spike_times[state][i] = current_summary["output_spike_times"]

            status.finish()
            status.start()

    count_sent_denominator = 1 if args.count_cpu_jobs_to_batch is None else args.count_cpu_jobs_to_batch

    try:
        job_factory = status.job_factory
        logger.info("going for join_all... (from last run)")
        job_factory_state = job_factory.join_all(on_join)
        if job_factory_state.is_successfull():
            logger.info("join_all (from last run) finished successfully!")
        else:
            err_msg = "join_all (from last run) finished unsuccessfully with job_factory_state {}".format(job_factory_state)
            logger.error(err_msg)
            if not args.retry:
                status.finish()
                return
    except:
        dataset_jobs_folder = os.path.join(simulation_dataset_folder, "dataset_jobs")
        os.makedirs(dataset_jobs_folder, exist_ok=True)
        job_factory = SlurmJobFactory(dataset_jobs_folder, args.count_cpu_jobs_to_batch,
                                       use_local=args.use_local, max_local_workers=args.max_local_workers)
        status.job_factory = job_factory
        status.save()

    status.finish()

    simulation_args_saver = get_simulation_args()


    while not status.all_finished():
        status.start()
        count_currently_sent = 0
        for state in ["train", "valid", "test"]:
            state_folder = os.path.join(simulation_dataset_folder, state)
            os.makedirs(state_folder, exist_ok=True)
            count_simulations = kwargs["count_simulations_for_{}".format(state)]
            for i in range(count_simulations):
                if not status.finished[state][i]:
                    simulate_args = copy.deepcopy(args)

                    exc_fr_options = {k: [args.exc_fr_per_synapse_options[2*k],args.exc_fr_per_synapse_options[2*k+1]] for k in range(len(args.exc_fr_per_synapse_options) // 2)}
                    inh_fr_options = {k: [args.inh_fr_per_synapse_options[2*k],args.inh_fr_per_synapse_options[2*k+1]] for k in range(len(args.inh_fr_per_synapse_options) // 2)}

                    if args.fr_option_choice_by_modulo:
                        fr_option_key = i % len(list(exc_fr_options.keys()))
                    else:
                        fr_option_key = np.random.choice(list(exc_fr_options.keys()))

                    low_range_exc, high_range_exc = exc_fr_options[fr_option_key]
                    low_range_inh, high_range_inh = inh_fr_options[fr_option_key]

                    count_exc_spikes_per_synapse_per_100ms_high = 0.1 * high_range_exc
                    count_exc_spikes_per_synapse_per_100ms_low = 0.1 * low_range_exc
                    count_inh_spikes_per_synapse_per_100ms_high = 0.1 * high_range_inh
                    count_inh_spikes_per_synapse_per_100ms_low = 0.1 * low_range_inh

                    simulate_args.count_exc_spikes_per_synapse_per_100ms_range = [count_exc_spikes_per_synapse_per_100ms_low, count_exc_spikes_per_synapse_per_100ms_high]
                    if np.random.rand() < args.same_exc_inh_count_spikes_per_synapse_per_100ms_range_prob:
                        simulate_args.count_inh_spikes_per_synapse_per_100ms_range = simulate_args.count_exc_spikes_per_synapse_per_100ms_range
                    else:
                        simulate_args.count_inh_spikes_per_synapse_per_100ms_range = [count_inh_spikes_per_synapse_per_100ms_low, count_inh_spikes_per_synapse_per_100ms_high]

                    exc_weight_ratio_options = {k: [args.exc_weight_ratio_options[2*k],args.exc_weight_ratio_options[2*k+1]] for k in range(len(args.exc_weight_ratio_options) // 2)}
                    inh_weight_ratio_options = {k: [args.inh_weight_ratio_options[2*k],args.inh_weight_ratio_options[2*k+1]] for k in range(len(args.inh_weight_ratio_options) // 2)}

                    weight_ratio_option_key = np.random.choice(list(exc_weight_ratio_options.keys()))

                    exc_weights_ratio_low, exc_weights_ratio_high = exc_weight_ratio_options[weight_ratio_option_key]
                    inh_weights_ratio_low, inh_weights_ratio_high = inh_weight_ratio_options[weight_ratio_option_key]                   

                    simulate_args.exc_weights_ratio_range = [exc_weights_ratio_low, exc_weights_ratio_high]
                    if np.random.rand() < args.same_exc_inh_weight_ratio_range_prob:
                        simulate_args.inh_weights_ratio_range = simulate_args.exc_weights_ratio_range
                    else:
                        simulate_args.inh_weights_ratio_range = [inh_weights_ratio_low, inh_weights_ratio_high]
                    
                    exc_initial_synapses_per_super_synapse_mult_factor_low = min(args.count_exc_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max[2], abs(np.random.normal(args.count_exc_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max[0], args.count_exc_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max[1])))                    
                    exc_initial_synapses_per_super_synapse_mult_factor_high = max(args.count_exc_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min[2], abs(np.random.normal(args.count_exc_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min[0], args.count_exc_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min[1])))
                    inh_initial_synapses_per_super_synapse_mult_factor_low = min(args.count_inh_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max[2], abs(np.random.normal(args.count_inh_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max[0], args.count_inh_initial_synapses_per_super_synapse_mult_factor_low_avg_std_max[1])))                    
                    inh_initial_synapses_per_super_synapse_mult_factor_high = max(args.count_inh_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min[2], abs(np.random.normal(args.count_inh_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min[0], args.count_inh_initial_synapses_per_super_synapse_mult_factor_high_avg_std_min[1])))

                    simulate_args.count_exc_initial_synapses_per_super_synapse_mult_factor_range = [exc_initial_synapses_per_super_synapse_mult_factor_low, exc_initial_synapses_per_super_synapse_mult_factor_high]
                    if np.random.rand() < args.same_exc_inh_initial_synapses_per_super_synapse_mult_factor_range_prob:
                        simulate_args.count_inh_initial_synapses_per_super_synapse_mult_factor_range = simulate_args.count_exc_initial_synapses_per_super_synapse_mult_factor_range
                    else:
                        simulate_args.count_inh_initial_synapses_per_super_synapse_mult_factor_range = [inh_initial_synapses_per_super_synapse_mult_factor_low, inh_initial_synapses_per_super_synapse_mult_factor_high]

                    simulate_args.random_seed = status.random_seed[state][i]
                    simulation_folder = os.path.join(state_folder, f"simulation_{i}")
                    simulation_arg_string = args_to_arg_string(simulate_args, list(simulation_args_saver.arguments.keys()))
                    simulation_save_plots = args.save_plots and not args.no_save_plots_for_simulations

                    if state == "train" and args.train_input_folder:
                        input_file = f'{args.train_input_folder}/{i}/all_weighted_spikes.npz'
                        input_file_arg_string = f"--input_file {input_file}"
                    elif state == "valid" and args.valid_input_folder:
                        input_file = f'{args.valid_input_folder}/{i}/all_weighted_spikes.npz'
                        input_file_arg_string = f"--input_file {input_file}"
                    elif state == "test" and args.test_input_folder:
                        input_file = f'{args.test_input_folder}/{i}/all_weighted_spikes.npz'
                        input_file_arg_string = f"--input_file {input_file}"
                    else:
                        input_file_arg_string = ""

                    job_name = f"run_simulation_{simulation_dataset_basename}_{state}_{i}"

                    extra_arg_string = extra_arg_string_func(args)
                    
                    job_factory.send_job(job_name,\
                        f"python3 -u {file_to_run} {simulation_arg_string} {input_file_arg_string} {extra_arg_string} --simulation_folder {simulation_folder} --save_plots {simulation_save_plots}",
                         extra=(state, i),
                          mem=args.cpu_job_memory*1000, use_scontrol=args.use_scontrol,
                           use_finishfile=args.use_finishfile, timelimit=args.simulation_job_timelimit)
                    count_currently_sent += 1
                    if count_currently_sent / count_sent_denominator >= args.maximum_cpu_parallel_jobs:
                        break
            if count_currently_sent / count_sent_denominator >= args.maximum_cpu_parallel_jobs:
                        break

        job_factory.flush()
        
        status.finish()
        status.start()

        logger.info("going for join_all...")
        job_factory_state = job_factory.join_all(on_join)

        if job_factory_state.is_successfull():
            logger.info("join_all finished successfully!")
            status.finish()
        else:
            err_msg = "join_all finished unsuccessfully with job_factory_state {}".format(job_factory_state)
            logger.error(err_msg)
            status.finish()
            if not args.retry:
                return

    logger.info("Going to calculate firing_rate_information and average_somatic_voltage_information...")
    calculation_start_time = time.time()

    firing_rate_information = {}
    isi_information = {}
    average_somatic_voltage_information = {}
    input_average_spikes_per_super_synapse_per_second_information = {}
    average_initial_neuron_weight_information = {}
    output_spike_times_information = {}

    for state in ["train", "valid", "test"]:
        current_firing_rates = {}
        current_isi = {}
        current_input_average_exc_spikes_per_super_synapse_per_second = {}
        current_input_average_inh_spikes_per_super_synapse_per_second = {}
        current_input_average_weighted_exc_spikes_per_super_synapse_per_second = {}
        current_input_average_weighted_inh_spikes_per_super_synapse_per_second = {}
        current_average_exc_initial_neuron_weight = {}
        current_average_inh_initial_neuron_weight = {}
        current_output_spike_times = {}
        average_somatic_voltage = 0.0
        average_clipped_somatic_voltage = 0.0
        count_simulations = kwargs["count_simulations_for_{}".format(state)]
        for i in range(count_simulations):
            current_firing_rates[i] = status.output_firing_rate_after_initialization[state][i]
            current_isi[i] = status.output_isi_after_initialization[state][i]
            current_input_average_exc_spikes_per_super_synapse_per_second[i] = status.input_average_exc_spikes_per_super_synapse_per_second[state][i]
            current_input_average_inh_spikes_per_super_synapse_per_second[i] = status.input_average_inh_spikes_per_super_synapse_per_second[state][i]
            current_input_average_weighted_exc_spikes_per_super_synapse_per_second[i] = status.input_average_weighted_exc_spikes_per_super_synapse_per_second[state][i]
            current_input_average_weighted_inh_spikes_per_super_synapse_per_second[i] = status.input_average_weighted_inh_spikes_per_super_synapse_per_second[state][i]

            current_average_exc_initial_neuron_weight[i] = status.average_exc_initial_neuron_weight[state][i]
            current_average_inh_initial_neuron_weight[i] = status.average_inh_initial_neuron_weight[state][i]

            if args.summarize_output_spike_times:
                current_output_spike_times[i] = status.output_spike_times[state][i]

            average_somatic_voltage += status.average_somatic_voltage[state][i]
            average_clipped_somatic_voltage += status.average_clipped_somatic_voltage[state][i]

        firing_rate_values = np.array(list(current_firing_rates.values()))
        if len(firing_rate_values) == 0:
            firing_rate_values = np.array([0.0])

        firing_rate_information[state] = {
            "by_index": current_firing_rates,
            "min": np.min(firing_rate_values),
            "avg": np.mean(firing_rate_values),
            "std": np.std(firing_rate_values),
            "max": np.max(firing_rate_values),
            "med": np.median(firing_rate_values)
        }

        isi_values = np.concatenate(list(current_isi.values()))
        if len(isi_values) == 0:
            isi_values = np.array([0.0])

        isi_information[state] = {
            "by_index": current_isi,
            "min": np.min(isi_values),
            "avg": np.mean(isi_values),
            "std": np.std(isi_values),
            "max": np.max(isi_values),
            "med": np.median(isi_values)
        }

        input_average_exc_spikes_per_super_synapse_per_second_values = np.array(list(current_input_average_exc_spikes_per_super_synapse_per_second.values()))
        if len(input_average_exc_spikes_per_super_synapse_per_second_values) == 0:
            input_average_exc_spikes_per_super_synapse_per_second_values = np.array([0.0])

        input_average_exc_spikes_per_super_synapse_per_second_information = {
            "by_index": current_input_average_exc_spikes_per_super_synapse_per_second,
            "min": np.min(input_average_exc_spikes_per_super_synapse_per_second_values),
            "avg": np.mean(input_average_exc_spikes_per_super_synapse_per_second_values),
            "std": np.std(input_average_exc_spikes_per_super_synapse_per_second_values),
            "max": np.max(input_average_exc_spikes_per_super_synapse_per_second_values),
            "med": np.median(input_average_exc_spikes_per_super_synapse_per_second_values)
        }

        input_average_inh_spikes_per_super_synapse_per_second_values = np.array(list(current_input_average_inh_spikes_per_super_synapse_per_second.values()))
        if len(input_average_inh_spikes_per_super_synapse_per_second_values) == 0:
            input_average_inh_spikes_per_super_synapse_per_second_values = np.array([0.0])

        input_average_inh_spikes_per_super_synapse_per_second_information = {
            "by_index": current_input_average_inh_spikes_per_super_synapse_per_second,
            "min": np.min(input_average_inh_spikes_per_super_synapse_per_second_values),
            "avg": np.mean(input_average_inh_spikes_per_super_synapse_per_second_values),
            "std": np.std(input_average_inh_spikes_per_super_synapse_per_second_values),
            "max": np.max(input_average_inh_spikes_per_super_synapse_per_second_values),
            "med": np.median(input_average_inh_spikes_per_super_synapse_per_second_values)
        }

        input_average_weighted_exc_spikes_per_super_synapse_per_second_values = np.array(list(current_input_average_weighted_exc_spikes_per_super_synapse_per_second.values()))
        if len(input_average_weighted_exc_spikes_per_super_synapse_per_second_values) == 0:
            input_average_weighted_exc_spikes_per_super_synapse_per_second_values = np.array([0.0])

        input_average_weighted_exc_spikes_per_super_synapse_per_second_information = {
            "by_index": current_input_average_weighted_exc_spikes_per_super_synapse_per_second,
            "min": np.min(input_average_weighted_exc_spikes_per_super_synapse_per_second_values),
            "avg": np.mean(input_average_weighted_exc_spikes_per_super_synapse_per_second_values),
            "std": np.std(input_average_weighted_exc_spikes_per_super_synapse_per_second_values),
            "max": np.max(input_average_weighted_exc_spikes_per_super_synapse_per_second_values),
            "med": np.median(input_average_weighted_exc_spikes_per_super_synapse_per_second_values)
        }

        input_average_weighted_inh_spikes_per_super_synapse_per_second_values = np.array(list(current_input_average_weighted_inh_spikes_per_super_synapse_per_second.values()))
        if len(input_average_weighted_inh_spikes_per_super_synapse_per_second_values) == 0:
            input_average_weighted_inh_spikes_per_super_synapse_per_second_values = np.array([0.0])

        input_average_weighted_inh_spikes_per_super_synapse_per_second_information = {
            "by_index": current_input_average_weighted_inh_spikes_per_super_synapse_per_second,
            "min": np.min(input_average_weighted_inh_spikes_per_super_synapse_per_second_values),
            "avg": np.mean(input_average_weighted_inh_spikes_per_super_synapse_per_second_values),
            "std": np.std(input_average_weighted_inh_spikes_per_super_synapse_per_second_values),
            "max": np.max(input_average_weighted_inh_spikes_per_super_synapse_per_second_values),
            "med": np.median(input_average_weighted_inh_spikes_per_super_synapse_per_second_values)
        }
        input_average_spikes_per_super_synapse_per_second_information[state] = {"exc": input_average_exc_spikes_per_super_synapse_per_second_information, "inh": input_average_inh_spikes_per_super_synapse_per_second_information,
        "weighted_exc": input_average_weighted_exc_spikes_per_super_synapse_per_second_information, "weighted_inh": input_average_weighted_inh_spikes_per_super_synapse_per_second_information}

        average_exc_initial_neuron_weight_values = np.array(list(current_average_exc_initial_neuron_weight.values()))
        if len(average_exc_initial_neuron_weight_values) == 0:
            average_exc_initial_neuron_weight_values = np.array([0.0])

        average_exc_initial_neuron_weight_information = {
            "by_index": current_average_exc_initial_neuron_weight,
            "min": np.min(average_exc_initial_neuron_weight_values),
            "avg": np.mean(average_exc_initial_neuron_weight_values),
            "std": np.std(average_exc_initial_neuron_weight_values),
            "max": np.max(average_exc_initial_neuron_weight_values),
            "med": np.median(average_exc_initial_neuron_weight_values)
        }

        average_inh_initial_neuron_weight_values = np.array(list(current_average_inh_initial_neuron_weight.values()))
        if len(average_inh_initial_neuron_weight_values) == 0:
            average_inh_initial_neuron_weight_values = np.array([0.0])

        average_inh_initial_neuron_weight_information = {
            "by_index": current_average_inh_initial_neuron_weight,
            "min": np.min(average_inh_initial_neuron_weight_values),
            "avg": np.mean(average_inh_initial_neuron_weight_values),
            "std": np.std(average_inh_initial_neuron_weight_values),
            "max": np.max(average_inh_initial_neuron_weight_values),
            "med": np.median(average_inh_initial_neuron_weight_values)
        }

        average_initial_neuron_weight_information[state] = {"exc": average_exc_initial_neuron_weight_information, "inh": average_inh_initial_neuron_weight_information}

        if args.summarize_output_spike_times:
            output_spike_times_information[state] = current_output_spike_times
        else:
            output_spike_times_information[state] = []

        average_somatic_voltage = average_somatic_voltage / count_simulations if count_simulations > 0 else 0.0
        average_clipped_somatic_voltage = average_clipped_somatic_voltage / count_simulations if count_simulations > 0 else 0.0
        average_somatic_voltage_information[state] = {"average_somatic_voltage": average_somatic_voltage, "average_clipped_somatic_voltage": average_clipped_somatic_voltage}

        fr_info_for_log = {k:v for k, v in firing_rate_information[state].items() if k != "by_index"}
        isi_info_for_log = {k:v for k, v in isi_information[state].items() if k != "by_index"}

        exc_average_spikes_info_for_log = {k:v for k, v in input_average_spikes_per_super_synapse_per_second_information[state]["exc"].items() if k != "by_index"}
        inh_average_spikes_info_for_log = {k:v for k, v in input_average_spikes_per_super_synapse_per_second_information[state]["inh"].items() if k != "by_index"}

        weighted_exc_average_spikes_info_for_log = {k:v for k, v in input_average_spikes_per_super_synapse_per_second_information[state]["weighted_exc"].items() if k != "by_index"}
        weighted_inh_average_spikes_info_for_log = {k:v for k, v in input_average_spikes_per_super_synapse_per_second_information[state]["weighted_inh"].items() if k != "by_index"}

        average_exc_initial_neuron_weight_information_info_for_log = {k:v for k, v in average_initial_neuron_weight_information[state]["exc"].items() if k != "by_index"}
        average_inh_initial_neuron_weight_information_info_for_log = {k:v for k, v in average_initial_neuron_weight_information[state]["inh"].items() if k != "by_index"}

        logger.info(f"{state}:")
        logger.info(f"firing rate information for {state} is {fr_info_for_log}")
        logger.info(f"isi information for {state} is {isi_info_for_log}")
        logger.info(f"input average exc spikes per super synapse per second information for {state} is {exc_average_spikes_info_for_log}")
        logger.info(f"input average inh spikes per super synapse per second information for {state} is {inh_average_spikes_info_for_log}")
        logger.info(f"input average weighted exc spikes per super synapse per second information for {state} is {weighted_exc_average_spikes_info_for_log}")
        logger.info(f"input average weighted inh spikes per super synapse per second information for {state} is {weighted_inh_average_spikes_info_for_log}")
        logger.info(f"initial exc neurons weights information for {state} is {average_exc_initial_neuron_weight_information_info_for_log}")
        logger.info(f"initial inh neurons weights information for {state} is {average_inh_initial_neuron_weight_information_info_for_log}")
        logger.info(f"average somatic voltage for {state} is {average_somatic_voltage_information[state]}")

        if args.save_plots:
            plt.hist(firing_rate_values, bins=10)
            plt.savefig(f'{simulation_dataset_folder}/{state}/firing_rate_hist.png')
            plt.close('all')

    calculation_duration_in_seconds = time.time() - calculation_start_time
    logger.info(f"calculate firing_rate_information, isi_information, average_somatic_voltage_information, input_average_spikes_per_super_synapse_per_second_information and average_initial_neuron_weight_information finished!, it took {calculation_duration_in_seconds:.3f} seconds")

    valid_fr_and_isi_by_index = {}
    for index in firing_rate_information['valid']['by_index']:
        valid_fr_and_isi_by_index[index] = {'fr':firing_rate_information['valid']['by_index'][index], 'isi':isi_information['valid']['by_index'][index]}
    normalized_valid_set_by_index, normalized_valid_average_firing_rate, normalized_valid_average_isi = choose_normalized_fr_isi_set(valid_fr_and_isi_by_index, kwargs, 'valid')
    firing_rate_information['valid']['normalized_valid_set_by_index'] = normalized_valid_set_by_index
    firing_rate_information['valid']['normalized_valid_average_firing_rate'] = normalized_valid_average_firing_rate
    isi_information['valid']['normalized_valid_average_isi'] = normalized_valid_average_isi

    test_fr_and_isi_by_index = {}
    for index in firing_rate_information['test']['by_index']:
        test_fr_and_isi_by_index[index] = {'fr':firing_rate_information['test']['by_index'][index], 'isi':isi_information['test']['by_index'][index]}
    normalized_test_set_by_index, normalized_test_average_firing_rate, normalized_test_average_isi = choose_normalized_fr_isi_set(test_fr_and_isi_by_index, kwargs, 'test')
    firing_rate_information['test']['normalized_test_set_by_index'] = normalized_test_set_by_index
    firing_rate_information['test']['normalized_test_average_firing_rate'] = normalized_test_average_firing_rate
    isi_information['test']['normalized_test_average_isi'] = normalized_test_average_isi

    logger.info(f"train set has {(args.count_simulations_for_train*(status.simulation_duration_in_ms))/1000/60/60:.3f} hours of data")
    logger.info(f"valid set has {(args.count_simulations_for_valid*(status.simulation_duration_in_ms))/1000/60/60:.3f} hours of data")
    logger.info(f"normalized valid set has {(args.count_simulations_for_normalized_valid*(status.simulation_duration_in_ms))/1000/60/60:.3f} hours of data")
    logger.info(f"test set has {(args.count_simulations_for_test*(status.simulation_duration_in_ms))/1000/60/60:.3f} hours of data")
    logger.info(f"normalized test set has {(args.count_simulations_for_normalized_test*(status.simulation_duration_in_ms))/1000/60/60:.3f} hours of data")

    pickle.dump({'args':args, 'firing_rate_information': firing_rate_information, 'isi_information': isi_information, 'average_somatic_voltage_information': average_somatic_voltage_information, 'input_average_spikes_per_super_synapse_per_second_information': input_average_spikes_per_super_synapse_per_second_information, 'average_initial_neuron_weight_information': average_initial_neuron_weight_information, 'output_spike_times_information': output_spike_times_information}, open(f'{simulation_dataset_folder}/summary.pkl', 'wb'), protocol=-1)

    pickle.dump({'args':args, 'count_simulations': args.count_simulations_for_train, 'firing_rate_information': firing_rate_information['train'], 'isi_information': isi_information['train'], 'average_somatic_voltage_information': average_somatic_voltage_information['train'], 'input_average_spikes_per_super_synapse_per_second_information': input_average_spikes_per_super_synapse_per_second_information['train'], 'average_initial_neuron_weight_information':average_initial_neuron_weight_information['train'], 'output_spike_times_information': output_spike_times_information['train']}, open(f'{simulation_dataset_folder}/train/summary.pkl', 'wb'), protocol=-1)
    pickle.dump({'args':args, 'count_simulations': args.count_simulations_for_valid, 'firing_rate_information': firing_rate_information['valid'], 'isi_information': isi_information['valid'], 'average_somatic_voltage_information': average_somatic_voltage_information['valid'], 'input_average_spikes_per_super_synapse_per_second_information': input_average_spikes_per_super_synapse_per_second_information['valid'], 'average_initial_neuron_weight_information':average_initial_neuron_weight_information['valid'], 'output_spike_times_information': output_spike_times_information['valid']}, open(f'{simulation_dataset_folder}/valid/summary.pkl', 'wb'), protocol=-1)
    pickle.dump({'args':args, 'count_simulations': args.count_simulations_for_test, 'firing_rate_information': firing_rate_information['test'], 'isi_information': isi_information['test'], 'average_somatic_voltage_information': average_somatic_voltage_information['test'], 'input_average_spikes_per_super_synapse_per_second_information': input_average_spikes_per_super_synapse_per_second_information['test'], 'average_initial_neuron_weight_information':average_initial_neuron_weight_information['test'], 'output_spike_times_information': output_spike_times_information['test']}, open(f'{simulation_dataset_folder}/test/summary.pkl', 'wb'), protocol=-1)

    generate_dataset_total_duration_in_seconds = status.get_duration()

    logger.info(f"generate dataset finished!, it took {generate_dataset_total_duration_in_seconds/60.0:.3f} minutes")

    return generate_dataset_total_duration_in_seconds

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())
    
    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to generate dataset for neuron! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    generate_dataset(args)
    logger.info(f"Goodbye from generate dataset for neuron! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
