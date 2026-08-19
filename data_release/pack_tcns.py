"""
Pack the trained networks of one model for release.

    python data_release/pack_tcns.py --model Rat_L5b_PC_2_Hay_passive_dends_simple_soma --output_folder /path/to/tcns

For each of the three seeds of the depth-3 network of the paper it writes

    /path/to/tcns/models/<model>/d3_w128_t<seed>/models/model_<epoch>_<steps>   best checkpoint
    /path/to/tcns/models/<model>/d3_w128_t<seed>/models/model_*_light.pkl       every light checkpoint
    /path/to/tcns/models/<model>/d3_w128_t<seed>/args.pkl, final_results.pkl
    /path/to/tcns/models/<model>/results.csv, final_results.pkl                  all 12 nets of the model

i.e. the layout of a trained network folder, as `train_neuron_tcn.py` writes it, so
`calculate_fci.py --neuron_tcn_folder .../d3_w128_t<seed>` works on it.

The best checkpoint is the one `best_save` of the last light checkpoint points at, i.e. the
network whose AUC the paper reports. It is re-saved with torch.save under the same name and
the same keys as the original, minus the optimizer state and with the weights cast to float32
(the originals are float64 with Adam state, 355 MB; this is 59 MB), so `calculate_fci.py`,
`Tcn.get_tcn_from_file` and `load_saved_models(..., load_best_save=True)` read it unchanged.
Loading float32 weights into the (float64) Tcn is a plain cast; the outputs differ from the
original by ~1e-7.

Everything else is copied as is.
"""

import os
import sys
import glob
import shutil
import pickle
import logging
import argparse

import torch

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_FOLDER)

from data_release import models as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, choices=M.release_names(), metavar='MODEL')
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--keep_float64', action='store_true', help='do not cast the weights to float32')
    return parser.parse_args()


def latest_light_checkpoint(models_folder):
    lights = glob.glob(os.path.join(models_folder, 'model_*_light.pkl'))
    if not lights:
        raise FileNotFoundError(f"no light checkpoints in {models_folder}")
    return max(lights, key=lambda f: int(os.path.basename(f).split('_')[2]))


def slim_checkpoint(checkpoint, keep_float64=False):
    slim = {k: v for k, v in checkpoint.items() if k != 'optimizer_state_dict'}
    state_dict = checkpoint['model_state_dict']
    if not keep_float64:
        state_dict = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in state_dict.items()}
    slim['model_state_dict'] = state_dict
    slim['release_note'] = ('re-saved for release: optimizer state removed' +
                            ('' if keep_float64 else ', weights cast from float64 to float32'))
    return slim


def pack_net(src_net_folder, dst_net_folder, keep_float64=False):
    src_models = os.path.join(src_net_folder, 'models')
    dst_models = os.path.join(dst_net_folder, 'models')
    os.makedirs(dst_models, exist_ok=True)

    for f in glob.glob(os.path.join(src_models, 'model_*_light.pkl')):
        dst = os.path.join(dst_models, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copyfile(f, dst)
    for name in ('args.pkl', 'final_results.pkl'):
        src = os.path.join(src_net_folder, name)
        if os.path.exists(src) and not os.path.exists(os.path.join(dst_net_folder, name)):
            shutil.copyfile(src, os.path.join(dst_net_folder, name))

    best_save = pickle.load(open(latest_light_checkpoint(src_models), 'rb'))['best_save']
    best_name = f"model_{best_save['ep']}_{best_save['steps']}"
    src_best = os.path.join(src_models, best_name)
    dst_best = os.path.join(dst_models, best_name)
    if not os.path.exists(src_best):
        logger.warning(f"{src_best} does not exist (pruned?) - only the light checkpoints are shipped for this seed")
        return dict(best=best_name, exists=False, auc=best_save.get('auc'), normalized_auc=best_save.get('normalized_auc'))
    if not os.path.exists(dst_best):
        checkpoint = torch.load(src_best, map_location='cpu')
        tmp = f'{dst_best}.{os.getpid()}.tmp'
        torch.save(slim_checkpoint(checkpoint, keep_float64), tmp)
        os.replace(tmp, dst_best)
    logger.info(f"{dst_best}: {os.path.getsize(dst_best)/1e6:.1f} MB, auc {best_save.get('auc'):.5f}")
    return dict(best=best_name, exists=True, auc=best_save.get('auc'), normalized_auc=best_save.get('normalized_auc'))


def main():
    args = get_args()
    src = M.neuron_tcns_folder(args.model)
    dst = os.path.join(args.output_folder, 'models', args.model)
    os.makedirs(dst, exist_ok=True)
    for name in ('results.csv', 'final_results.pkl'):
        if os.path.exists(os.path.join(src, name)) and not os.path.exists(os.path.join(dst, name)):
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    for seed in M.SEEDS:
        info = pack_net(os.path.join(src, M.TCN_NAME.format(seed=seed)),
                        os.path.join(dst, M.TCN_RELEASE_NAME.format(seed=seed)), args.keep_float64)
        logger.info(f"{args.model} seed {seed}: {info}")


if __name__ == '__main__':
    main()
