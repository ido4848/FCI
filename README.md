# FCI: Functional Complexity Index

This repository contains the code accompanying the preprint:

**["What makes human cortical pyramidal neurons functionally complex"](https://www.biorxiv.org/content/10.1101/2024.12.17.628883v1)**

Ido Aizenbud, Daniela Yoeli, David Beniaguev, Christiaan PJ de Kock, Michael London, Idan Segev

## Abstract
Humans exhibit unique cognitive abilities within the animal kingdom, but the neural mechanisms driving these advanced capabilities remain poorly understood. Human cortical neurons differ from those of other species, such as rodents, in both their morphological and physiological characteristics. Could the distinct properties of human cortical neurons help explain the superior cognitive capabilities of humans? Understanding this relationship requires a metric to quantify how neuronal properties contribute to the functional complexity of single neurons, yet no such standardized measure currently exists. Here, we propose the Functional Complexity Index (FCI), a generalized, deep learning-based framework to assess the input-output complexity of neurons. By comparing the FCI of cortical pyramidal neurons from different layers in rats and humans, we identified key morpho-electrical factors that underlie functional complexity. Human cortical pyramidal neurons were found to be significantly more functionally complex than their rat counterparts, primarily due to differences in dendritic membrane area and branching pattern, as well as density and nonlinearity of NMDA-mediated synaptic receptors. These findings reveal the structural-biophysical basis for the enhanced functional properties of human neurons.

![](fig1.png)

---

## Usage

The FCI pipeline consists of three main steps:
1. **Create a simulation dataset** for a neuron model
2. **Train a TCN** (Temporal Convolutional Network) on the dataset
3. **Calculate the FCI** from the trained model

### Prerequisites

- Python 3.8+
- NEURON simulator
- PyTorch
- SLURM (optional - for cluster job submission; use `--use_local` flag for local execution)

Install dependencies:
```bash
pip install numpy matplotlib torch scikit-learn h5py wandb
```

---

### Step 1: Create a Dataset for a Neuron Model

#### 1.1 Compile the NEURON model

First, navigate to the neuron model folder and compile the MOD files:

```bash
cd simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay_passive_dends_simple_soma
nrnivmodl mods
```

This will create an architecture-specific folder (e.g., `x86_64/`) containing the compiled mechanisms.

#### 1.2 Generate the simulation dataset

Use the `submit_simulate_neuron_and_create_dataset.py` script to generate training, validation, and test datasets. This script submits SLURM jobs to run neuron simulations in parallel.

```bash
python simulating_neurons/submit_simulate_neuron_and_create_dataset.py \
    --neuron_model_folder simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay_passive_dends_simple_soma \
    --simulation_dataset_folder /path/to/output/dataset \
    --simulation_dataset_name my_neuron_dataset \
    --count_simulations_for_train 100 \
    --count_simulations_for_valid 20 \
    --count_simulations_for_test 20
```

**Running without SLURM (local execution):**

If you don't have access to a SLURM cluster, you can run simulations locally using the `--use_local` flag:

```bash
python simulating_neurons/submit_simulate_neuron_and_create_dataset.py \
    --neuron_model_folder simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay_passive_dends_simple_soma \
    --simulation_dataset_folder /path/to/output/dataset \
    --simulation_dataset_name my_neuron_dataset \
    --count_simulations_for_train 20 \
    --count_simulations_for_valid 5 \
    --count_simulations_for_test 5 \
    --use_local \
    --max_local_workers 4
```

**Key parameters:**
- `--neuron_model_folder`: Path to the compiled neuron model
- `--simulation_dataset_folder`: Output folder for the dataset
- `--simulation_dataset_name`: Name for the dataset
- `--count_simulations_for_train`: Number of training simulations (default: 20)
- `--count_simulations_for_valid`: Number of validation simulations (default: 10)
- `--count_simulations_for_test`: Number of test simulations (default: 10)
- `--use_local`: Run jobs locally instead of using SLURM (default: False)
- `--max_local_workers`: Maximum parallel workers for local execution (default: CPU count - 1)

The script will create `train/`, `valid/`, and `test/` subdirectories containing the simulation data.

---

### Step 2: Train a TCN on the Dataset

Once the dataset is generated, train a Temporal Convolutional Network (TCN) to predict the neuron's output spikes from its synaptic inputs:

```bash
python training_nets/train_neuron_tcn.py \
    --simulation_dataset_folder /path/to/output/dataset/my_neuron_dataset \
    --neuron_tcn_folder /path/to/output/tcn \
    --neuron_tcn_name my_neuron_tcn
```

**Key parameters:**
- `--simulation_dataset_folder`: Path to the dataset created in Step 1
- `--neuron_tcn_folder`: Output folder for the trained model
- `--neuron_tcn_name`: Name for the trained TCN

The training will log metrics including AUC (Area Under the ROC Curve), which is used to calculate the FCI.

---

### Step 3: Calculate the Functional Complexity Index (FCI)

After training, calculate the FCI using the `calculate_fci.py` script:

```bash
python calculate_fci.py --neuron_tcn_folder /path/to/output/tcn/my_neuron_tcn
```

**Options:**
```bash
# Calculate FCI from a trained TCN folder
python calculate_fci.py --neuron_tcn_folder /path/to/neuron_tcn

# Calculate FCI from a specific results file
python calculate_fci.py --results_pkl /path/to/model_X_Y_test_results.pkl

# Calculate FCI from a specific AUC value
python calculate_fci.py --auc 0.95

# Use normalized AUC (from normalized firing rate test set)
python calculate_fci.py --neuron_tcn_folder /path/to/neuron_tcn --use_normalized
```

#### FCI Formula

The FCI is calculated from the AUC as:

$$\text{FCI} = \frac{\log_{10}(1000 \times (1 - \text{AUC}))}{\log_{10}(1000 \times (1 - 0.9))}$$

Where:
- AUC = 0.9 → FCI = 1.0 (maximum complexity)
- AUC = 0.999 → FCI ≈ 0.0 (near-perfect prediction, low complexity)

