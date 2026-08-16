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
    morph_name = "dend-C300797C-P4_axon-C200897C-P4_-_Scale_x1.000_y1.025_z1.000_-_Clone_11.asc"

    passive_dends_simple_soma_model_filename = path + "../../../passive_dends_simple_soma_model.hoc"

    h.load_file(passive_dends_simple_soma_model_filename)

    cell = h.PassiveDendsSimpleSomaModel(morph_dir, morph_name, 20.179) # 1040

    # from simulating_neurons/neuron_models/rat/hay/Rat_L5b_PC_2_Hay_passive
    rho_exemplar = 32.58690275984611
    rho_axon_exemplar = 247.23080159409773

    # from me
    rho_test = 32.26482228562258
    rho_axon_test = 189.1197104939959

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

    syn_df = create_synapses(cell, 'rat')

    logger.info(f"Created model with {len(syn_df['segments'])} segments")
    logger.info(f"Temperature is {h.celsius} degrees celsius")
    
    return cell, syn_df
