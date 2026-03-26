__all__ = ["Data_Set", "load_datasets"]

import os
import socket

import yaml


class Data_Set:
    """Container for a single beam-measurement dataset."""

    def __init__(self, pathlist, screen, save_loc, paths, computer, empty,
                 prefixes, DAQ_Matching, bg_file, raw_vcc):
        self.pathlist = pathlist
        self.screen = screen
        if computer == 'NERSC':
            path = paths['NERSC']
        elif computer == 's3df':
            path = paths['s3df']
        self.save_loc = path + save_loc
        self.empty = empty
        self.prefixes = prefixes
        self.DAQ_Matching = DAQ_Matching
        self.bg_file = bg_file
        self.raw_vcc = raw_vcc

    def return_params(self):
        return (self.pathlist, self.screen, self.save_loc, self.empty,
                self.prefixes, self.DAQ_Matching, self.bg_file, self.raw_vcc)


# ── helpers ────────────────────────────────────────────────────────────────

def _detect_computer():
    """Return 's3df' or 'NERSC' based on the current hostname."""
    return 's3df' if 'sdf' in socket.gethostname() else 'NERSC'


def load_datasets(yaml_path):
    """Read a *datasets.yaml* file and return a ``dict[str, Data_Set]``.

    Parameters
    ----------
    yaml_path : str
        Path to the datasets YAML file.  Copy
        ``config/example_datasets.yaml`` as a starting point.
    """
    with open(yaml_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Empty config → return an empty dict.
    if not cfg:
        return {}

    paths = cfg["paths"]
    computer = _detect_computer()
    empty = {k: "" for k in cfg["empty_keys"]}

    result = {}
    for name, entry in cfg["datasets"].items():
        result[name] = Data_Set(
            pathlist=entry["pathlist"],
            screen=entry["screen"],
            save_loc=entry["save_loc"],
            paths=paths,
            computer=computer,
            empty=empty,
            prefixes=entry.get("prefixes"),
            DAQ_Matching=entry.get("DAQ_Matching"),
            bg_file=entry.get("bg_file"),
            raw_vcc=entry.get("raw_vcc"),
        )

    # aliases – extra dict keys that point to an already-built Data_Set
    for alias, target in cfg.get("aliases", {}).items():
        result[alias] = result[target]

    return result
