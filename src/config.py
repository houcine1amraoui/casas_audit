"""Config precedence: defaults.yaml < config.yaml < --config file < --set a.b=c."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def deep_update(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def _load(p):
    p = Path(p)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    return out


def load_config(project_root_dir, dataset_folder, extra_config=None, sets=None):
    cfg = _load(REPO / "configs" / "defaults.yaml")
    deep_update(cfg, _load(REPO / "configs" / "config.yaml"))
    if extra_config:
        deep_update(cfg, _load(extra_config))
    deep_update(cfg, parse_set(sets))
    cfg["project_root_dir"] = str(project_root_dir)
    cfg["dataset_folder"] = str(dataset_folder)
    root = Path(project_root_dir)
    cfg["paths"]["output_dir"] = cfg["paths"]["output_dir"] or str(root / "outputs" / cfg["run_name"])
    cfg["paths"]["cache_dir"] = cfg["paths"]["cache_dir"] or str(root / "cache" / "casas")
    return cfg
