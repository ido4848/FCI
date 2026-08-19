#!/bin/bash
# Pack the 24 models of the paper for release, on the SLURM cluster.
#
#   bash data_release/submit_pack.sh              # submit everything that is not done yet
#   ONLY=Rat_L6_UPC_BBP_Mandge_passive_dends_simple_soma bash data_release/submit_pack.sh   # one model (canary)
#   DRY=1 bash data_release/submit_pack.sh        # print what would be submitted
#
# One job per (model, split) packs the simulations into shards and verifies them against the
# originals; one job per model packs the networks. A unit whose VERIFIED.ok / PACKED.ok marker
# exists, or whose job is already queued or running, is skipped, so the script can be re-run
# until everything is done. Runs from the workstation (hops to the login server for sbatch) or
# from the login server itself.

set -euo pipefail

# site-specific settings, all overridable: the python with this repo's dependencies, where to
# stage the packed release, which SLURM partition to use, and FCI_PIPELINES_ROOT (see models.py)
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
RELEASE=${RELEASE:-$REPO/FCI_release}
PARTITION=${PARTITION:-elscn.q}
ONLY=${ONLY:-}
DRY=${DRY:-}

if ! command -v sbatch >/dev/null 2>&1; then
    exec ssh -o BatchMode=yes loginserver "FCI_PIPELINES_ROOT='${FCI_PIPELINES_ROOT:-}' ONLY='$ONLY' DRY='$DRY' PARTITION='$PARTITION' RELEASE='$RELEASE' PY='$PY' bash $REPO/data_release/submit_pack.sh"
fi

mkdir -p "$RELEASE/dataset/models" "$RELEASE/tcns" "$RELEASE/logs" "$RELEASE/jobs"

MODELS=$($PY -c "import sys; sys.path.insert(0,'$REPO'); from data_release import models as M; print(' '.join(M.release_names()))")

# job names already in the queue (pending or running), so a re-run never submits a unit twice
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
queued() { grep -qxF "$1" <<< "$QUEUED"; }

submit() {  # submit <jobfile> -> prints job id
    if [ -n "$DRY" ]; then echo "DRY sbatch $1" >&2; echo "0"; return; fi
    local out
    out=$(sbatch "$1") || { echo "sbatch failed for $1: $out" >&2; exit 1; }
    echo "$out" | awk '{print $NF}'
}

n_sub=0
for model in $MODELS; do
    if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then continue; fi
    SDF=$($PY -c "import sys; sys.path.insert(0,'$REPO'); from data_release import models as M; print(M.simulation_dataset_folder('$model'))")
    OUT="$RELEASE/dataset/models/$model"

    for split in train test; do
        if [ -f "$OUT/$split/VERIFIED.ok" ]; then continue; fi
        name="pack_${split}_${model}"
        if queued "$name"; then echo "queued already   pack $model $split"; continue; fi
        job="$RELEASE/jobs/pack_${model}_${split}.job"
        cat > "$job" <<EOF
#!/bin/bash
#SBATCH --job-name $name
#SBATCH -o $RELEASE/logs/pack_${model}_${split}.out
#SBATCH -p $PARTITION
#SBATCH -c 4
#SBATCH --mem 6000
#SBATCH -t 0-08:00:00
set -euo pipefail
cd $REPO
$PY data_release/pack_dataset.py --simulation_dataset_folder $SDF --split $split --output_folder $OUT --workers 4 --model_name $model
$PY data_release/verify_pack.py --simulation_dataset_folder $SDF --split $split --packed_folder $OUT --sample 60 && touch $OUT/$split/VERIFIED.ok
EOF
        id=$(submit "$job"); echo "submitted $id  pack $model $split"; n_sub=$((n_sub+1))
    done

    if [ ! -f "$RELEASE/tcns/models/$model/PACKED.ok" ]; then
        name="tcns_${model}"
        if queued "$name"; then echo "queued already   tcns $model"; continue; fi
        job="$RELEASE/jobs/tcns_${model}.job"
        cat > "$job" <<EOF
#!/bin/bash
#SBATCH --job-name $name
#SBATCH -o $RELEASE/logs/tcns_${model}.out
#SBATCH -p $PARTITION
#SBATCH -c 1
#SBATCH --mem 8000
#SBATCH -t 0-02:00:00
set -euo pipefail
export FCI_PIPELINES_ROOT='${FCI_PIPELINES_ROOT:-}'
cd $REPO
$PY data_release/pack_tcns.py --model $model --output_folder $RELEASE/tcns && touch $RELEASE/tcns/models/$model/PACKED.ok
EOF
        id=$(submit "$job"); echo "submitted $id  tcns $model"; n_sub=$((n_sub+1))
    fi
done
echo "$n_sub jobs submitted"
