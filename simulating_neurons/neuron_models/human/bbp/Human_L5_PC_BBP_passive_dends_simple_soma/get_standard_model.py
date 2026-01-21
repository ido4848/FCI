from neuron import h,gui
import pandas as pd
import os
import logging
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.parent.parent.absolute()))

from simulating_neurons.neuron_models.model_utils import create_synapses

logger = logging.getLogger(__name__)

def create_cell(path=None):
    import neuron
    h.load_file("import3d.hoc")
    h.load_file("nrngui.hoc")

    if path is None:
        path = os.path.dirname(os.path.realpath(__file__)) +'/'
    
    neuron.load_mechanisms(path)

    morph_dir = path + "/morphologies/"

    morph_name = "2057_H21_29_197_11_01_03_metcontour.asc"

    passive_dends_simple_soma_model_filename = path + "../../../passive_dends_simple_soma_model.hoc"

    h.load_file(passive_dends_simple_soma_model_filename)

    cell = h.PassiveDendsSimpleSomaModel(morph_dir, morph_name, 53.1) # 1041

    # from neuron_models/rat/hay/Rat_L5b_PC_2_Hay_passive_dends_simple_soma
    rho_exemplar = 32.58690275984611
    rho_axon_exemplar = 247.23080159409773

    # from me
    rho_test = 48.177290634113085
    rho_axon_test = 462.9423383849466

    scale_factor = rho_test / rho_exemplar
    scale_factor_axon = rho_axon_test / rho_axon_exemplar

    # scaling soma conductances
    for sec in cell.all:
        if not ('soma' in sec.name()):
            continue
        sec.gbar_kv *= scale_factor
        sec.gbar_na *= scale_factor

    # scaling axon conductances
    for sec in cell.all:
        if not ('axon' in sec.name()):
            continue
        sec.gbar_kv *= scale_factor_axon
        sec.gbar_na *= scale_factor_axon

    syn_df = create_synapses(cell, 'human')

    logger.info(f"Created model with {len(syn_df['segments'])} segments")
    logger.info(f"Temperature is {h.celsius} degrees celsius")
    
    return cell, syn_df
