"""
The 24 models of the paper, and where their simulations and trained networks live.

`PIPELINES` maps the release name of every model - its folder name under
`simulating_neurons/neuron_models/` - to the internal pipeline folder that produced the paper's
numbers for it. Four models were renamed when the repository was published (their internal
folder lacks the `_passive_dends_simple_soma` suffix, or has a different index), which is why
this map exists at all instead of a naming rule.
"""

import os
import glob

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURON_MODELS_FOLDER = os.path.join(REPO_FOLDER, 'simulating_neurons', 'neuron_models')

# where the internal pipelines live. Only the machine that produced the paper's data has this;
# everything the download path needs (release_names, LABELS, ...) works without it.
PIPELINES_ROOT = os.environ.get('FCI_PIPELINES_ROOT', '/path/to/neuron_as_deep_net2_data')

# the network configuration of figure 2: depth 3, width 128, three seeds
TCN_NAME = 'neuron_tcn_d3_w128_fk54_lk12_sd0_5_bs180_lr0_01_wd1e_08_vlw0_02_t{seed}'
TCN_RELEASE_NAME = 'd3_w128_t{seed}'
SEEDS = (0, 1, 2)

# release name -> internal pipeline folder (order = pipeline_names_all in fig2.ipynb)
PIPELINES = {
    # rat, rat synapses
    'Rat_L2_TPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma': 'Rat_L2_TPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L23_PC_cADpyr229_1_BBP_passive_dends_simple_soma': 'Rat_L23_PC_cADpyr229_1_BBP_passive_standard0823_simple_pipeline_1',
    'Rat_L23_PC_cADpyr229_5_BBP_passive_dends_simple_soma': 'Rat_L23_PC_cADpyr229_5_BBP_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L4_TPC_BBP_Mandge_passive_dends_simple_soma': 'Rat_L4_TPC_BBP_Mandge_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L4_PC_cADpyr230_1_BBP_passive_dends_simple_soma': 'Rat_L4_PC_cADpyr230_1_BBP_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L4_PC_cADpyr230_2_BBP_passive_dends_simple_soma': 'Rat_L4_PC_cADpyr230_2_BBP_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L5_TPC_BBP_Mandge_passive_dends_simple_soma': 'Rat_L5_TPC_BBP_Mandge_passive_standard0823_simple_pipeline_1',
    'Rat_L5_TTPC1_cADpyr232_1_BBP_diams_fixed_passive_dends_simple_soma': 'Rat_L5_TTPC1_cADpyr232_1_BBP_diams_fixed_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L5b_PC_2_Hay_passive_dends_simple_soma': 'Rat_L5b_PC_2_Hay_passive_standard0823_simple_pipeline_1',
    'Rat_L6_UPC_BBP_Mandge_passive_dends_simple_soma': 'Rat_L6_UPC_BBP_Mandge_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L6_TPC_BBP_Mandge_passive_dends_simple_soma': 'Rat_L6_TPC_BBP_Mandge_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Rat_L6_IPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma': 'Rat_L6_IPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    # human, human synapses
    'Human_L23_PC_0603_11_937_Eyal_passive_dends_simple_soma': 'Human_L23_PC_0603_11_937_Eyal_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L23_PC_1303_03_448_Eyal_passive_dends_simple_soma': 'Human_L23_PC_1303_03_448_Eyal_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L3_PC_0_BBP_passive_dends_simple_soma': 'Human_L3_PC_0_BBP_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L4_PC_BBP_Mandge_diams_fixed_passive_dends_simple_soma': 'Human_L4_PC_BBP_Mandge_diams_fixed_passive_dends_simple_soma_standard0823_simple_pipeline_2',
    'Human_L4_PC_539661667_Allen_passive_dends_simple_soma': 'Human_L4_PC_539661667_Allen_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L4_PC_569818704_Allen_passive_dends_simple_soma': 'Human_L4_PC_569818704_Allen_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L5_PC_BBP_Mandge_passive_dends_simple_soma': 'Human_L5_PC_BBP_Mandge_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L5_PC_0_BBP_passive_dends_simple_soma': 'Human_L5_PC_1_BBP_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L5_PC_790872626_Allen_passive_dends_simple_soma': 'Human_L5_PC_790872626_Allen_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L6_PC_558211203_Allen_passive_dends_simple_soma': 'Human_L6_PC_558211203_Allen_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L6_PC_548494556_Allen_passive_dends_simple_soma': 'Human_L6_PC_548494556_Allen_passive_dends_simple_soma_standard0823_simple_pipeline_1',
    'Human_L6_PC_528614014_Allen_passive_dends_simple_soma': 'Human_L6_PC_528614014_Allen_passive_dends_simple_soma_standard0823_simple_pipeline_1',
}

# the short labels of figure 2
LABELS = {
    'Rat_L2_TPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma': 'Rat L2',
    'Rat_L23_PC_cADpyr229_1_BBP_passive_dends_simple_soma': 'Rat L2/3 (1)',
    'Rat_L23_PC_cADpyr229_5_BBP_passive_dends_simple_soma': 'Rat L2/3 (5)',
    'Rat_L4_TPC_BBP_Mandge_passive_dends_simple_soma': 'Rat L4 TPC',
    'Rat_L4_PC_cADpyr230_1_BBP_passive_dends_simple_soma': 'Rat L4 (1)',
    'Rat_L4_PC_cADpyr230_2_BBP_passive_dends_simple_soma': 'Rat L4 (2)',
    'Rat_L5_TPC_BBP_Mandge_passive_dends_simple_soma': 'Rat L5 TPC',
    'Rat_L5_TTPC1_cADpyr232_1_BBP_diams_fixed_passive_dends_simple_soma': 'Rat L5 TTPC',
    'Rat_L5b_PC_2_Hay_passive_dends_simple_soma': 'Rat L5b Hay',
    'Rat_L6_UPC_BBP_Mandge_passive_dends_simple_soma': 'Rat L6 UPC',
    'Rat_L6_TPC_BBP_Mandge_passive_dends_simple_soma': 'Rat L6 TPC',
    'Rat_L6_IPC_BBP_Mandge_diams_fixed_passive_dends_simple_soma': 'Rat L6 IPC',
    'Human_L23_PC_0603_11_937_Eyal_passive_dends_simple_soma': 'Human L2/3 (937)',
    'Human_L23_PC_1303_03_448_Eyal_passive_dends_simple_soma': 'Human L2/3 (448)',
    'Human_L3_PC_0_BBP_passive_dends_simple_soma': 'Human L3',
    'Human_L4_PC_BBP_Mandge_diams_fixed_passive_dends_simple_soma': 'Human L4 (1496)',
    'Human_L4_PC_539661667_Allen_passive_dends_simple_soma': 'Human L4 (539)',
    'Human_L4_PC_569818704_Allen_passive_dends_simple_soma': 'Human L4 (569)',
    'Human_L5_PC_BBP_Mandge_passive_dends_simple_soma': 'Human L5 (1833)',
    'Human_L5_PC_0_BBP_passive_dends_simple_soma': 'Human L5 (2057)',
    'Human_L5_PC_790872626_Allen_passive_dends_simple_soma': 'Human L5 (790)',
    'Human_L6_PC_558211203_Allen_passive_dends_simple_soma': 'Human L6 (558)',
    'Human_L6_PC_548494556_Allen_passive_dends_simple_soma': 'Human L6 (548)',
    'Human_L6_PC_528614014_Allen_passive_dends_simple_soma': 'Human L6 (528)',
}

SPLITS = ('train', 'test')


def release_names():
    return list(PIPELINES)


def pipeline_folder(release_name):
    return os.path.join(PIPELINES_ROOT, PIPELINES[release_name])


def simulation_dataset_folder(release_name):
    return os.path.join(pipeline_folder(release_name), 'simulation_dataset')


def neuron_tcns_folder(release_name):
    return os.path.join(pipeline_folder(release_name), 'neuron_tcns')


def neuron_model_folder(release_name):
    """The model folder inside this repository (e.g. .../rat/hay/Rat_L5b_PC_2_Hay_passive_dends_simple_soma)."""
    hits = glob.glob(os.path.join(NEURON_MODELS_FOLDER, '*', '*', release_name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{release_name}: expected one model folder, found {hits}")
    return hits[0]


def species(release_name):
    return 'human' if release_name.startswith('Human') else 'rat'


def source(release_name):
    """eyal / allen / hay / bbp, from the model folder's parent."""
    return os.path.basename(os.path.dirname(neuron_model_folder(release_name)))


def layer(release_name):
    """L2, L2/3, L4, ... from the release name."""
    token = release_name.split('_')[1]
    return {'L23': 'L2/3', 'L5b': 'L5b'}.get(token, token)
