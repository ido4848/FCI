"""
Run the whole FCI pipeline, small and fast, to check that this repository works end to end.

It runs the four stages on a single neuron model, with tiny settings throughout:

    1. create an io matrix              (a handful of simulations, a 2x2 grid of input rates)
    2. generate a simulation dataset    (a handful of simulations, using the io matrix that the
                                         model ships - i.e. the one used in the paper)
    3. train a TCN                      (a small network, a few hundred steps, cpu is fine)
    4. calculate the FCI                (from the AUC of that network)

None of the numbers it produces are meaningful - the point is that every stage runs, reads what
the previous stage wrote, and produces the files the next stage expects. For the real settings
that the paper used see the README.

Before running, compile the MOD files of the model:

    cd simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay_passive_dends_simple_soma
    nrnivmodl mods

Usage:
    python quick_test.py
    python quick_test.py --neuron_model_folder <other model> --output_folder <where to write>
    python quick_test.py --stages dataset train fci      # skip the slow io matrix stage
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess

REPO_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_FOLDER)

from simulating_neurons.io_matrix_utils import load_legitimate_options

DEFAULT_NEURON_MODEL_FOLDER = os.path.join(REPO_FOLDER, "simulating_neurons", "neuron_models",
                                           "rat", "hay", "Rat_L5b_PC_2_Hay_passive_dends_simple_soma")

STAGES = ["io_matrix", "dataset", "train", "fci"]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--neuron_model_folder', default=DEFAULT_NEURON_MODEL_FOLDER)
    parser.add_argument('--output_folder', default=os.path.join(REPO_FOLDER, "quick_test_output"))
    parser.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    parser.add_argument('--max_local_workers', default=4, type=int)
    parser.add_argument('--simulation_duration_in_seconds', default=6, type=int)
    parser.add_argument('--count_simulations_for_train', default=8, type=int)
    parser.add_argument('--count_simulations_for_valid', default=2, type=int)
    parser.add_argument('--count_simulations_for_test', default=4, type=int)
    parser.add_argument('--maximum_train_steps', default=200, type=int)
    parser.add_argument('--clean', action='store_true', help='delete --output_folder before running')
    return parser.parse_args()


def run_stage(name, command):
    print(f"\n{'=' * 100}\n[quick_test] {name}\n{'=' * 100}")
    print(" ".join(command) + "\n", flush=True)

    start_time = time.time()
    process = subprocess.run(command, cwd=REPO_FOLDER)
    duration = time.time() - start_time

    if process.returncode != 0:
        print(f"\n[quick_test] {name} FAILED after {duration / 60:.1f} minutes "
              f"(exit code {process.returncode})")
        sys.exit(process.returncode)

    print(f"\n[quick_test] {name} took {duration / 60:.1f} minutes")
    return duration


def get_quick_search_ranges(neuron_model_folder):
    """
    A narrow input rate window to search in, so the io matrix stage stays short.

    The real calibration starts from zero input and walks up until it finds inputs that make the
    neuron fire at ~1 Hz, which takes hundreds of simulations. Here we start the search at the
    rates the model is already known to respond to, and accept almost any output firing rate, so
    it terminates after a single iteration.
    """
    io_matrix = load_legitimate_options(neuron_model_folder)
    if io_matrix is None:
        return [0, 20], [0, 20]

    first_option = io_matrix["options"][0]
    exc_low = first_option["exc_fr_per_synapse_range"][0]
    inh_low = first_option["inh_fr_per_synapse_range"][0]

    return [exc_low, exc_low + 1.0], [inh_low, inh_low + 1.0]


def main():
    args = get_args()

    neuron_model_folder = os.path.abspath(args.neuron_model_folder)
    output_folder = os.path.abspath(args.output_folder)

    if args.clean and os.path.exists(output_folder):
        print(f"[quick_test] deleting {output_folder}")
        shutil.rmtree(output_folder)

    io_matrix_folder = os.path.join(output_folder, "io_matrix")
    simulation_dataset_folder = os.path.join(output_folder, "simulation_dataset")
    neuron_tcn_folder = os.path.join(output_folder, "neuron_tcn")

    compiled = [f for f in os.listdir(neuron_model_folder) if f.startswith(("x86_64", "arm64", "aarch64"))]
    if not compiled:
        print(f"[quick_test] {neuron_model_folder} has no compiled mechanisms.\n"
              f"             run `cd {neuron_model_folder} && nrnivmodl mods` first.")
        sys.exit(1)

    print(f"[quick_test] neuron model: {neuron_model_folder}")
    print(f"[quick_test] output:       {output_folder}")
    print(f"[quick_test] stages:       {', '.join(args.stages)}")

    # wandb is optional for training, and the quick test should not wait on the network for it
    os.environ.setdefault("WANDB_MODE", "disabled")

    common_simulation_arguments = [
        "--neuron_model_folder", neuron_model_folder,
        "--simulation_duration_in_seconds", str(args.simulation_duration_in_seconds),
        "--use_local",
        "--max_local_workers", str(args.max_local_workers),
        "--save_plots", "False",
    ]

    durations = {}

    if "io_matrix" in args.stages:
        exc_range, inh_range = get_quick_search_ranges(neuron_model_folder)
        durations["io_matrix"] = run_stage("1/4 create a small io matrix", [
            sys.executable, os.path.join("simulating_neurons", "create_io_matrix.py"),
            "--io_matrix_folder", io_matrix_folder,
            *common_simulation_arguments,
            "--count_simulations_for_option", "2",
            "--count_options_per_axis", "2",
            "--count_simulations_for_train", "2",
            "--count_simulations_for_test", "2",
            "--count_simulations_for_valid", "2",
            "--exc_fr_per_synapse_option_range", str(exc_range[0]), str(exc_range[1]),
            "--inh_fr_per_synapse_option_range", str(inh_range[0]), str(inh_range[1]),
            "--exc_fr_per_synapse_bin_size", "0.5",
            "--inh_fr_per_synapse_bin_size", "0.5",
            # accept nearly any output firing rate, so that one iteration is enough
            "--legitimate_output_fr_range", "0.01", "100.0",
            "--legitimate_options_count", "1",
        ])

        with open(os.path.join(io_matrix_folder, "legitimate_options.json")) as f:
            print(f"[quick_test] io matrix options found: {json.load(f)['options']}")

    if "dataset" in args.stages:
        # no --exc/inh_fr_per_synapse_options here on purpose: they are taken from the io matrix
        # that the model ships, which is the one the paper used
        durations["dataset"] = run_stage("2/4 generate a small simulation dataset", [
            sys.executable, os.path.join("simulating_neurons", "submit_simulate_neuron_and_create_dataset.py"),
            "--simulation_dataset_folder", simulation_dataset_folder,
            "--simulation_dataset_name", "quick_test_dataset",
            *common_simulation_arguments,
            "--count_simulations_for_train", str(args.count_simulations_for_train),
            "--count_simulations_for_valid", str(args.count_simulations_for_valid),
            "--count_simulations_for_test", str(args.count_simulations_for_test),
        ])

    if "train" in args.stages:
        durations["train"] = run_stage("3/4 train a small TCN", [
            sys.executable, os.path.join("training_nets", "train_neuron_tcn.py"),
            "--simulation_dataset_folder", simulation_dataset_folder,
            "--neuron_tcn_folder", neuron_tcn_folder,
            "--neuron_tcn_name", "quick_test_tcn",
            "--depth", "1",
            "--width", "8",
            "--batch_size", "4",
            "--maximum_train_steps", str(args.maximum_train_steps),
            "--test_every_in_steps", str(max(1, args.maximum_train_steps // 2)),
            "--test_normalized_every_in_steps", str(max(1, args.maximum_train_steps // 2)),
            "--show_train_results_every_in_steps", str(max(1, args.maximum_train_steps // 2)),
            "--test_normalized_every_in_count_tests", "1",
            "--count_train_workers", "1",
            "--count_test_workers", "1",
            "--count_normalized_test_workers", "1",
            "--use_data_parallel", "False",
            "--run_on_gpu", "False",
            "--save_plots", "False",
        ])

    if "fci" in args.stages:
        durations["fci"] = run_stage("4/4 calculate the FCI", [
            # the paper scores the normalized AUC (--use_normalized), which needs a test set big
            # enough to draw a 1 Hz subset out of - not the case for this handful of simulations
            sys.executable, "calculate_fci.py",
            "--neuron_tcn_folder", neuron_tcn_folder,
        ])

    print(f"\n{'=' * 100}")
    print("[quick_test] all stages passed")
    for stage, duration in durations.items():
        print(f"[quick_test]   {stage:<10} {duration / 60:6.1f} minutes")
    print(f"[quick_test] output is in {output_folder}")
    print("[quick_test] the FCI above is meaningless - it comes from a network trained for "
          f"{args.maximum_train_steps} steps on {args.count_simulations_for_train} simulations")


if __name__ == "__main__":
    main()
