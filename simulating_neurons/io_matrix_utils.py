"""
Reading and writing the input-output (io) matrix of a neuron model.

The io matrix maps a pair of (excitatory, inhibitory) input firing rate ranges - in spikes per
synapse per second - onto the average output firing rate that the neuron model produces when it
is driven at those rates. The pairs whose average output firing rate falls inside
`--legitimate_output_fr_range` are the *legitimate options*, and they are what dataset generation
samples its input rates from. They are model specific: a model with a large dendritic membrane
area needs very different input rates than a small one in order to fire at the same output rate,
so without them the datasets of different models are not comparable and the resulting FCI values
are meaningless.

Every neuron model in this repository ships the io matrix that was used in the paper under
`<neuron_model_folder>/io_matrix/`, so datasets can be regenerated without repeating the
calibration. `create_io_matrix.py` recreates one from scratch for a new model.

Two files make up a shipped io matrix:

- `legitimate_options.json` - the calibrated options, plus the settings they were calibrated
  with. `exc_fr_per_synapse_options` / `inh_fr_per_synapse_options` hold them in the flat
  `low_0 high_0 low_1 high_1 ...` form that `submit_simulate_neuron_and_create_dataset.py`
  expects on the command line.
- `io_matrix.csv` - every (exc, inh) bin that was probed during the calibration and the average
  output firing rate it produced, i.e. the full measured io matrix.
"""

import os
import csv
import json
import logging

logger = logging.getLogger(__name__)

IO_MATRIX_FOLDER_NAME = "io_matrix"
LEGITIMATE_OPTIONS_FILENAME = "legitimate_options.json"
IO_MATRIX_FILENAME = "io_matrix.csv"

FR_UNITS = "spikes per synapse per second (Hz)"

IO_MATRIX_CSV_FIELDS = ["exc_fr_per_synapse_low", "exc_fr_per_synapse_high",
                        "inh_fr_per_synapse_low", "inh_fr_per_synapse_high",
                        "average_output_fr", "std_output_fr", "count_simulations", "legitimate"]


def get_io_matrix_folder_for_neuron_model_folder(neuron_model_folder):
    return os.path.join(neuron_model_folder, IO_MATRIX_FOLDER_NAME)


def flatten_options(options):
    """[((exc_low, exc_high), (inh_low, inh_high)), ...] -> two flat [low, high, low, high, ...] lists."""
    exc_flat = []
    inh_flat = []
    for exc_range, inh_range in options:
        exc_flat += [float(exc_range[0]), float(exc_range[1])]
        inh_flat += [float(inh_range[0]), float(inh_range[1])]
    return exc_flat, inh_flat


def unflatten_options(exc_flat, inh_flat):
    """The inverse of flatten_options."""
    return [((exc_flat[2 * k], exc_flat[2 * k + 1]), (inh_flat[2 * k], inh_flat[2 * k + 1]))
            for k in range(len(exc_flat) // 2)]


def save_legitimate_options(io_matrix_folder, legitimate_options, calibration=None,
                            average_output_frs=None, std_output_frs=None):
    """
    Write the calibrated options of a model to `<io_matrix_folder>/legitimate_options.json`.

    Parameters
    ----------
    io_matrix_folder : str
        Folder to write into (created if needed).
    legitimate_options : list
        [((exc_low, exc_high), (inh_low, inh_high)), ...], in spikes per synapse per second.
    calibration : dict, optional
        The settings the options were calibrated with (target output firing rate range, number of
        simulations per option, bin sizes, ...). Stored as-is for provenance.
    average_output_frs, std_output_frs : list, optional
        Mean / std output firing rate measured for each option, in Hz.
    """
    os.makedirs(io_matrix_folder, exist_ok=True)

    exc_flat, inh_flat = flatten_options(legitimate_options)

    options = []
    for k, (exc_range, inh_range) in enumerate(legitimate_options):
        option = {"exc_fr_per_synapse_range": [float(exc_range[0]), float(exc_range[1])],
                  "inh_fr_per_synapse_range": [float(inh_range[0]), float(inh_range[1])]}
        if average_output_frs is not None:
            option["average_output_fr"] = float(average_output_frs[k])
        if std_output_frs is not None:
            option["std_output_fr"] = float(std_output_frs[k])
        options.append(option)

    content = {
        "units": FR_UNITS,
        "count_options": len(options),
        "calibration": calibration if calibration is not None else {},
        "options": options,
        "exc_fr_per_synapse_options": exc_flat,
        "inh_fr_per_synapse_options": inh_flat,
    }

    path = os.path.join(io_matrix_folder, LEGITIMATE_OPTIONS_FILENAME)
    with open(path, "w") as f:
        json.dump(content, f, indent=4)
    logger.info(f"saved {len(options)} legitimate options to {path}")

    return path


def load_legitimate_options(neuron_model_folder):
    """
    Load the io matrix options shipped with a neuron model.

    Returns the parsed json, or None when the model does not ship an io matrix.
    """
    path = os.path.join(get_io_matrix_folder_for_neuron_model_folder(neuron_model_folder),
                        LEGITIMATE_OPTIONS_FILENAME)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        content = json.load(f)

    exc_flat = content.get("exc_fr_per_synapse_options")
    inh_flat = content.get("inh_fr_per_synapse_options")
    if not exc_flat or not inh_flat:
        raise ValueError(f"{path} has no exc_fr_per_synapse_options / inh_fr_per_synapse_options")
    if len(exc_flat) != len(inh_flat):
        raise ValueError(f"{path} has {len(exc_flat)} exc and {len(inh_flat)} inh values, must be equal")
    if len(exc_flat) % 2 != 0:
        raise ValueError(f"{path} has an odd number of values, must be pairs of (low, high)")

    return content


def save_io_matrix_csv(io_matrix_folder, rows, filename=IO_MATRIX_FILENAME):
    """
    Write the measured io matrix to `<io_matrix_folder>/io_matrix.csv`.

    `rows` is an iterable of dicts with the keys in IO_MATRIX_CSV_FIELDS.
    """
    os.makedirs(io_matrix_folder, exist_ok=True)
    path = os.path.join(io_matrix_folder, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=IO_MATRIX_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in IO_MATRIX_CSV_FIELDS})
    logger.info(f"saved io matrix with {len(rows)} probed options to {path}")

    return path


def io_matrix_rows_from_df(df):
    """Turn the dataframe built by create_io_matrix.py into io_matrix.csv rows."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "exc_fr_per_synapse_low": row.exc_range[0],
            "exc_fr_per_synapse_high": row.exc_range[1],
            "inh_fr_per_synapse_low": row.inh_range[0],
            "inh_fr_per_synapse_high": row.inh_range[1],
            "average_output_fr": round(float(row.average_output_fr), 6),
            "std_output_fr": round(float(row.std_output_fr), 6),
            "count_simulations": len(row.output_fr),
            "legitimate": bool(row.legitimate),
        })
    return rows
