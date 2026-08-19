"""
Write models.json and the README.md (dataset card / model card) of the two release folders.

    python data_release/make_cards.py --release_folder /path/to/FCI_release [--namespace <hf user or org>]

reads the packed release (data_release/submit_pack.sh output) plus the internal pipelines for
the calibrated input rates and the reported AUC of every model, and writes

    <release_folder>/dataset/models.json
    <release_folder>/dataset/README.md
    <release_folder>/tcns/README.md
"""

import os
import sys
import json
import glob
import pickle
import argparse

import numpy as np

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_FOLDER)

from data_release import models as M  # noqa: E402
from calculate_fci import auc_to_fci  # noqa: E402

DEFAULT_NAMESPACE = os.environ.get('FCI_HF_NAMESPACE', 'i-do-ai')
DATASET_REPO_NAME = 'fci-neuron-simulations'
TCNS_REPO_NAME = 'fci-neuron-tcns'
LICENSE = 'cc-by-nc-nd-4.0'
# Fig. 2A of the paper reports, per model, the minimum over the three seeds of the FCI of the best
# normalized AUC among the evaluations made within the first 500,000 training steps
FIG2_MAX_STEP = 500000
PAPER_URL = 'https://www.pnas.org/doi/10.1073/pnas.2533168123'
CODE_URL = 'https://github.com/ido4848/FCI'

CITATION = """@article{aizenbud2026fci,
    title   = {Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons},
    author  = {Aizenbud, Ido and Yoeli, Daniela and Beniaguev, David and de Kock, Christiaan P. J. and London, Michael and Segev, Idan},
    journal = {Proceedings of the National Academy of Sciences},
    volume  = {123},
    number  = {28},
    pages   = {e2533168123},
    year    = {2026},
    doi     = {10.1073/pnas.2533168123}
}"""


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release_folder', required=True)
    parser.add_argument('--namespace', default=DEFAULT_NAMESPACE)
    return parser.parse_args()


def paper_auc(light_checkpoint):
    """(normalized AUC, step) the paper used for this seed: best evaluation at <= FIG2_MAX_STEP steps."""
    history = light_checkpoint['eval_history']['normalized_test']
    best = None
    for step, evaluation in history.items():
        if step <= FIG2_MAX_STEP and (best is None or evaluation['auc'] > best[0]):
            best = (evaluation['auc'], step)
    return best


def folder_size(folder, pattern='**/*'):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(folder, pattern), recursive=True) if os.path.isfile(f))


def count_simulations(split_folder):
    import h5py
    n = 0
    for path in glob.glob(os.path.join(split_folder, 'shard_*.h5')):
        with h5py.File(path, 'r') as h:
            n += len(h.attrs['simulation_indices'])
    return n


def model_record(name, release_folder):
    dataset_folder = os.path.join(release_folder, 'dataset', 'models', name)
    tcns_folder = os.path.join(release_folder, 'tcns', 'models', name)
    record = dict(name=name, label=M.LABELS[name], species=M.species(name), layer=M.layer(name), source=M.source(name),
                  neuron_model_folder=os.path.relpath(M.neuron_model_folder(name), REPO_FOLDER),
                  internal_pipeline=M.PIPELINES[name])

    summary = pickle.load(open(os.path.join(dataset_folder, 'dataset_summary.pkl'), 'rb'))
    rates = summary['input_average_spikes_per_super_synapse_per_second_information']['train']
    record['input_rate_hz_per_super_synapse'] = dict(
        exc=float(np.mean(list(rates['exc']['by_index'].values()))),
        inh=float(np.mean(list(rates['inh']['by_index'].values()))))
    record['output_firing_rate_hz'] = float(np.mean(list(summary['firing_rate_information']['train']['by_index'].values())))

    record['splits'] = {}
    for split in M.SPLITS:
        split_folder = os.path.join(dataset_folder, split)
        if os.path.isdir(split_folder):
            record['splits'][split] = dict(simulations=count_simulations(split_folder),
                                           shards=len(glob.glob(os.path.join(split_folder, 'shard_*.h5'))),
                                           bytes=folder_size(split_folder))

    record['tcns'] = {}
    for seed in M.SEEDS:
        net = os.path.join(tcns_folder, M.TCN_RELEASE_NAME.format(seed=seed))
        lights = glob.glob(os.path.join(net, 'models', 'model_*_light.pkl'))
        if not lights:
            continue
        light = pickle.load(open(max(lights, key=lambda f: int(os.path.basename(f).split('_')[2])), 'rb'))
        best = light['best_save']
        ckpt = os.path.join(net, 'models', f"model_{best['ep']}_{best['steps']}")
        paper_normalized_auc, paper_step = paper_auc(light)
        record['tcns'][M.TCN_RELEASE_NAME.format(seed=seed)] = dict(
            checkpoint=os.path.basename(ckpt) if os.path.exists(ckpt) else None,
            checkpoint_step=best.get('steps'),
            auc=best.get('auc'), normalized_auc=best.get('normalized_auc'),
            fci=float(auc_to_fci(best['auc'])) if best.get('auc') else None,
            normalized_fci=float(auc_to_fci(best['normalized_auc'])) if best.get('normalized_auc') else None,
            somatic_voltage_explained_variance=best.get('somatic_voltage_explained_variance'),
            paper_normalized_auc=paper_normalized_auc, paper_step=paper_step,
            paper_fci=float(auc_to_fci(paper_normalized_auc)))
    if record['tcns']:
        record['fig2_fci'] = min(t['paper_fci'] for t in record['tcns'].values())
    return record


def gb(n):
    return f"{n/1e9:.1f}"


def models_table(records, with_sizes=True):
    lines = ['| Model | Species | Layer | Source | Input rate exc / inh (Hz) | FCI (Fig. 2A) |' + (' Train | Test |' if with_sizes else ''),
             '| --- | --- | --- | --- | --- | --- |' + (' --- | --- |' if with_sizes else '')]
    for r in records:
        fci = f"{r['fig2_fci']:.4f}" if r.get('fig2_fci') is not None else '-'
        rate = r['input_rate_hz_per_super_synapse']
        row = f"| `{r['name']}` | {r['species']} | {r['layer']} | {r['source']} | {rate['exc']:.1f} / {rate['inh']:.1f} | {fci} |"
        if with_sizes:
            tr = r['splits'].get('train', {}); te = r['splits'].get('test', {})
            row += f" {gb(tr.get('bytes', 0))} GB | {gb(te.get('bytes', 0))} GB |"
        lines.append(row)
    return '\n'.join(lines)


def dataset_card(records, namespace):
    total = sum(s['bytes'] for r in records for s in r['splits'].values())
    n_train = sum(r['splits'].get('train', {}).get('simulations', 0) for r in records)
    n_test = sum(r['splits'].get('test', {}).get('simulations', 0) for r in records)
    return f"""---
license: {LICENSE}
pretty_name: FCI neuron simulations
tags:
- neuroscience
- computational-neuroscience
- biophysical-neuron-models
- dendrites
- spiking
- NEURON
- time-series
size_categories:
- 100K<n<1M
---

# FCI neuron simulations

The simulated input-output data of the 24 detailed compartmental neuron models compared in

**[Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons]({PAPER_URL})**
Ido Aizenbud, Daniela Yoeli, David Beniaguev, Christiaan P. J. de Kock, Michael London, Idan Segev.
*PNAS* 123(28), e2533168123 (2026).

Code, neuron models and instructions: **[{CODE_URL}]({CODE_URL})**. The trained networks
that the paper's Functional Complexity Index (FCI) values come from are in the companion model
repository [{namespace}/{TCNS_REPO_NAME}](https://huggingface.co/{namespace}/{TCNS_REPO_NAME}).

Every model was simulated with NEURON under in-vivo-like random synaptic input (Poisson trains
on ~1000 excitatory and ~1000 inhibitory "super synapses" spread over the dendrites, at input
rates calibrated per model so that the cell fires at ~1 Hz), and its somatic voltage and output
spikes were recorded. A deep network (TCN) trained on these simulations to predict the output
spikes from the input spikes gives an AUC, and the FCI is a fixed function of that AUC.

{n_train:,} training and {n_test:,} test simulations of 10.5 s each ({gb(total)} GB in total,
1-23 GB per model). 12 human models (human synaptic parameters) and 12 rat models (rat synaptic
parameters), pyramidal cells of layers 2-6, morphologies and biophysics from Eyal et al., the
Allen Cell Types Database, Hay et al. and the Blue Brain Project.

## Models

{models_table(records)}

Input rates are the mean over the simulations of the average firing rate per super synapse. The
FCI column is the value printed above each neuron in Fig. 2A of the paper (see the model
repository for how it follows from the three trained networks). `models.json` holds these
numbers, the sizes and the AUCs per seed in machine readable form.

## Layout

```
models/<model>/dataset_summary.pkl        summary of the whole dataset (calibrated input rates per simulation, ...)
models/<model>/train/summary.pkl          summary of the split, read by the trainer
models/<model>/train/shard_000.h5 ...     500 simulations per shard, 20 shards
models/<model>/test/summary.pkl
models/<model>/test/shard_000.h5 ...      5 shards
models.json
```

`<model>` is the model's folder name under `simulating_neurons/neuron_models/` in the code
repository, e.g. `Rat_L5b_PC_2_Hay_passive_dends_simple_soma`.

## Format of a shard

HDF5, one group per simulation:

| Dataset / attribute | dtype | shape | Content |
| --- | --- | --- | --- |
| `sim_<n>/exc_indptr` | uint32 | (segments+1,) | CSR row pointer of the excitatory input matrix (segments × ms) |
| `sim_<n>/exc_indices_delta` | uint16 | (nnz,) | column (ms) of every nonzero, delta coded within a row: the first nonzero of a row stores its column, every later one the distance from the previous nonzero |
| `sim_<n>/exc_data` | uint8 | (nnz,) | number of synaptic activations of that segment in that ms |
| `sim_<n>/inh_indptr`, `inh_indices_delta`, `inh_data` | | | the same for the inhibitory input |
| `sim_<n>/somatic_voltage` | float32 | (ms,) | somatic membrane potential, mV, 1 kHz |
| `sim_<n>/output_spike_times` | int64 | (spikes,) | ms |
| `sim_<n>.attrs` | | | `exc_shape`, `inh_shape`, output firing rate, input rates, average voltage, ... and `args_json` (the simulation's arguments) |
| file attrs | | | `model`, `split`, `shard`, `simulation_indices`, `format_version` |

The first 500 ms of every simulation are initialization and are skipped by the trainer. The
delta coding is lossless; the voltage was recorded in float64 and stored in float32.

```python
import h5py, numpy as np
from scipy import sparse

def read_csr(g, tag):
    indptr = g[f'{{tag}}_indptr'][:].astype(np.int64)
    delta = g[f'{{tag}}_indices_delta'][:].astype(np.int64)
    data = g[f'{{tag}}_data'][:]
    cumulative = np.cumsum(delta)
    starts = indptr[:-1]
    base = np.where(starts > 0, cumulative[np.maximum(starts - 1, 0)], 0)
    indices = cumulative - np.repeat(base, np.diff(indptr))
    return sparse.csr_matrix((data.astype(np.float64), indices, indptr), shape=tuple(g.attrs[f'{{tag}}_shape']))

with h5py.File('models/Rat_L5b_PC_2_Hay_passive_dends_simple_soma/test/shard_000.h5') as h:
    g = h['sim_0']
    exc = read_csr(g, 'exc').toarray()          # (1041, 10500)
    inh = read_csr(g, 'inh').toarray()
    v = g['somatic_voltage'][:]
    spikes = g['output_spike_times'][:]
```

`utils/simulation_shards.py` in the code repository has this reader (and the writer), and
`training_nets/train_neuron_tcn.py` trains on a downloaded model folder directly:

```bash
pip install huggingface_hub
python data_release/download.py --model Rat_L5b_PC_2_Hay_passive_dends_simple_soma --splits train test
python training_nets/train_neuron_tcn.py --simulation_dataset_folder paper_data/simulations/models/Rat_L5b_PC_2_Hay_passive_dends_simple_soma ...
```

## Citation

```bibtex
{CITATION}
```
"""


def tcns_card(records, namespace):
    lines = ['| Model | Seed | Checkpoint (step) | Normalized AUC | FCI | Paper: normalized AUC (step) | Paper: FCI | Fig. 2A |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in records:
        for net, t in r['tcns'].items():
            ckpt = f"{t['checkpoint']} ({t['checkpoint_step']:,})" if t['checkpoint'] else f"light checkpoints only ({t['checkpoint_step']:,})"
            fig2 = f"**{r['fig2_fci']:.4f}**" if abs(t['paper_fci'] - r['fig2_fci']) < 1e-12 else ''
            lines.append(f"| `{r['name']}` | {net} | {ckpt} | {t['normalized_auc']:.5f} | {t['normalized_fci']:.4f} | "
                         f"{t['paper_normalized_auc']:.5f} ({t['paper_step']:,}) | {t['paper_fci']:.4f} | {fig2} |")
    table = '\n'.join(lines)
    return f"""---
license: {LICENSE}
tags:
- neuroscience
- computational-neuroscience
- temporal-convolutional-network
- pytorch
library_name: pytorch
---

# FCI neuron TCNs

The trained networks behind the Functional Complexity Index values reported in

**[Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons]({PAPER_URL})**
Ido Aizenbud, Daniela Yoeli, David Beniaguev, Christiaan P. J. de Kock, Michael London, Idan Segev.
*PNAS* 123(28), e2533168123 (2026).

Code: **[{CODE_URL}]({CODE_URL})**. Training data: [{namespace}/{DATASET_REPO_NAME}](https://huggingface.co/datasets/{namespace}/{DATASET_REPO_NAME}).

For each of the 24 neuron models: the three seeds of the depth-3, width-128 temporal
convolutional network of the paper (`neuron_tcn_d3_w128_fk54_lk12_sd0_5_bs180_lr0_01_wd1e_08_vlw0_02_t{{0,1,2}}`
internally), trained to predict the neuron's output spikes and somatic voltage from its
synaptic input. FCI = log10(1000 (1 - AUC)) / log10(1000 (1 - 0.9)), from the AUC on the
normalized test set (a subset of the test simulations whose mean output rate is exactly 1 Hz).

## Layout

```
models/<model>/results.csv, final_results.pkl                        final AUCs of all 12 networks trained on the model (depths 1, 2, 3, 7 x 3 seeds)
models/<model>/d3_w128_t<seed>/args.pkl, final_results.pkl           the network's training arguments and final results
models/<model>/d3_w128_t<seed>/models/model_<epoch>_<steps>          the best checkpoint (torch.save)
models/<model>/d3_w128_t<seed>/models/model_*_light.pkl              every evaluation along training (best_save + eval_history, plain pickle)
```

`<model>` is the model's folder name under `simulating_neurons/neuron_models/` in the code
repository. A `d3_w128_t<seed>` folder has the layout `train_neuron_tcn.py` writes, so

```bash
python calculate_fci.py --neuron_tcn_folder models/<model>/d3_w128_t0 --use_normalized
```

prints the FCI of that seed's checkpoint. The checkpoint is a dict with `model_state_dict`, `args`,
`in_chans`, `best_save`, `eval_history`, `epoch`, `step`, ...; load it with
`Tcn.get_tcn_from_file(path, 'cpu')` from `training_nets/train_neuron_tcn.py`. Compared with the
checkpoints written during training, the optimizer state was removed and the weights were cast
from float64 to float32 (355 MB → 59 MB per network; the network runs in float64 and its outputs
change by ~1e-7). One checkpoint (`Rat_L6_IPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma`,
seed 1) was pruned before the release and only its light checkpoints, i.e. its AUC, survive.

## Which number is in the paper

Every network was evaluated on the normalized test set every ~50,000 steps and its
`model_*_light.pkl` files keep that history (`eval_history['normalized_test'][step]['auc']`).
The **checkpoint** shipped for a seed is `best_save`: the evaluation with the highest normalized
AUC over the whole run. The **paper** (Fig. 2A) took, per seed, the highest normalized AUC among
the evaluations made within the first {FIG2_MAX_STEP:,} training steps, and printed above each
neuron the *minimum* FCI over its three seeds - the "Fig. 2A" column below, which reproduces the
paper's 24 values exactly. Where a run kept improving after step {FIG2_MAX_STEP:,} the shipped
checkpoint is a little better than the paper's number for that seed.

```python
import pickle, glob, numpy as np
from calculate_fci import auc_to_fci
light = max(glob.glob('models/<model>/d3_w128_t0/models/model_*_light.pkl'), key=lambda f: int(f.split('_')[-3]))
history = pickle.load(open(light, 'rb'))['eval_history']['normalized_test']
paper_auc = max(v['auc'] for step, v in history.items() if step <= {FIG2_MAX_STEP})
print(auc_to_fci(paper_auc))
```

## Networks

{table}

## Citation

```bibtex
{CITATION}
```
"""


def main():
    args = get_args()
    records = [model_record(name, args.release_folder) for name in M.release_names()]
    dataset_folder = os.path.join(args.release_folder, 'dataset')
    tcns_folder = os.path.join(args.release_folder, 'tcns')
    json.dump(records, open(os.path.join(dataset_folder, 'models.json'), 'w'), indent=1)
    open(os.path.join(dataset_folder, 'README.md'), 'w').write(dataset_card(records, args.namespace))
    open(os.path.join(tcns_folder, 'README.md'), 'w').write(tcns_card(records, args.namespace))
    total = sum(s['bytes'] for r in records for s in r['splits'].values())
    print(f"{len(records)} models, {total/1e9:.1f} GB of simulations; wrote models.json and both README.md")


if __name__ == '__main__':
    main()
