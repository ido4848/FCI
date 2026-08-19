"""
Upload the packed release to Hugging Face, one model at a time, restartable.

    hf auth login                                     # once, with a write token
    python data_release/upload.py --release_folder /path/to/FCI_release --namespace <user or org> --dry_run
    python data_release/upload.py --release_folder /path/to/FCI_release --namespace <user or org>

Creates (private) `<namespace>/fci-neuron-simulations` (dataset) and `<namespace>/fci-neuron-tcns`
(model) if they do not exist, uploads README.md / models.json, then every `models/<model>`
folder as its own commit. A model whose files are already in the repo with the same sizes is
skipped, so the script can be re-run after an interruption and only finishes what is missing.
Make the repos public from their settings page (or with --public) once everything is up.

The upload runs wherever this is started; on the ELSC cluster the compute nodes have internet
access, so a long upload is best submitted as a job:

    sbatch -p elscn.q -c 4 --mem 8000 -t 3-00:00:00 -o upload.out --wrap \\
        "cd $PWD && <python> data_release/upload.py --release_folder ... --namespace ..."
"""

import os
import sys
import time
import glob
import argparse

REPO_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_FOLDER)

from data_release import models as M  # noqa: E402
from data_release.make_cards import DATASET_REPO_NAME, TCNS_REPO_NAME, DEFAULT_NAMESPACE  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release_folder', required=True)
    parser.add_argument('--namespace', default=DEFAULT_NAMESPACE)
    parser.add_argument('--what', nargs='*', default=['dataset', 'tcns'], choices=['dataset', 'tcns'])
    parser.add_argument('--models', nargs='*', default=None, choices=M.release_names(), metavar='MODEL')
    parser.add_argument('--public', action='store_true', help='create the repos public (default: private)')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def local_files(folder):
    """{relative path: size} of every file under folder."""
    out = {}
    for path in glob.glob(os.path.join(folder, '**', '*'), recursive=True):
        if os.path.isfile(path) and not path.endswith('.tmp') and not path.endswith('.ok'):
            out[os.path.relpath(path, folder)] = os.path.getsize(path)
    return out


def remote_files(api, repo_id, repo_type, path_in_repo):
    """{path relative to path_in_repo: size} of what the repo already holds there."""
    from huggingface_hub.utils import HfHubHTTPError
    out = {}
    try:
        for entry in api.list_repo_tree(repo_id, path_in_repo=path_in_repo, repo_type=repo_type, recursive=True):
            size = getattr(entry, 'size', None)
            if size is not None:
                out[os.path.relpath(entry.path, path_in_repo)] = size
    except HfHubHTTPError as e:  # the folder is not in the repo yet
        if e.response is None or e.response.status_code != 404:
            raise
    return out


def upload_release(api, args, kind):
    folder = os.path.join(args.release_folder, 'dataset' if kind == 'dataset' else 'tcns')
    repo_type = 'dataset' if kind == 'dataset' else 'model'
    repo_id = f"{args.namespace}/{DATASET_REPO_NAME if kind == 'dataset' else TCNS_REPO_NAME}"
    print(f"\n=== {repo_type} {repo_id} <- {folder}")

    if not args.dry_run:
        api.create_repo(repo_id, repo_type=repo_type, private=not args.public, exist_ok=True)

    # the small top level files first (card, index)
    for name in ('README.md', 'models.json'):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            print(f"  {name}")
            if not args.dry_run:
                api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo_id, repo_type=repo_type,
                                commit_message=f"add {name}")

    models = args.models or M.release_names()
    total = 0
    t0 = time.time()
    for name in models:
        model_folder = os.path.join(folder, 'models', name)
        if not os.path.isdir(model_folder):
            print(f"  {name}: not packed yet, skipping")
            continue
        markers = glob.glob(os.path.join(model_folder, '**', 'VERIFIED.ok'), recursive=True) + \
            glob.glob(os.path.join(model_folder, 'PACKED.ok'))
        expected_markers = len(M.SPLITS) if kind == 'dataset' else 1
        if len(markers) < expected_markers:
            print(f"  {name}: only {len(markers)}/{expected_markers} done markers, skipping")
            continue
        local = local_files(model_folder)
        size = sum(local.values())
        remote = remote_files(api, repo_id, repo_type, f'models/{name}') if not args.dry_run else {}
        if local and all(remote.get(k) == v for k, v in local.items()):
            print(f"  {name}: {size/1e9:.2f} GB already uploaded")
            continue
        print(f"  {name}: {len(local)} files, {size/1e9:.2f} GB ...", flush=True)
        if not args.dry_run:
            t = time.time()
            api.upload_folder(folder_path=model_folder, path_in_repo=f'models/{name}', repo_id=repo_id,
                              repo_type=repo_type, commit_message=f"add {name}", ignore_patterns=['*.tmp', '*.ok'])
            dt = time.time() - t
            print(f"      done in {dt/60:.1f} min ({size/1e6/dt:.1f} MB/s)")
        total += size
    print(f"=== {repo_id}: {total/1e9:.1f} GB {'to upload' if args.dry_run else 'uploaded'} in {(time.time()-t0)/60:.1f} min")


def main():
    args = get_args()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    api = HfApi()
    if not args.dry_run:
        who = api.whoami()
        print(f"logged in as {who['name']}")
    for kind in args.what:
        upload_release(api, args, kind)


if __name__ == '__main__':
    main()
