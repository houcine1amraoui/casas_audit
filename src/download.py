"""Download the CASAS free-living files from Zenodo into the dataset folder (Drive), verify md5, save record metadata."""
import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("ca")
BASE = "https://zenodo.org"


def md5sum(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(cfg):
    """Returns {file name: local path}. Never re-downloads a file whose md5 already matches."""
    import requests
    z = cfg["zenodo"]
    root = Path(cfg["dataset_folder"])
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    meta = root / f"zenodo_{z['record']}.json"
    if not meta.exists():
        try:
            r = requests.get(f"{BASE}/api/records/{z['record']}", timeout=60)
            r.raise_for_status()
            meta.write_text(json.dumps(r.json(), indent=1))
            log.info("saved record metadata -> %s", meta)
        except Exception as e:  # metadata is useful (home table) but not required
            log.warning("could not fetch record metadata: %s", e)
    for name, md5 in z["files"].items():
        p = root / name
        ok = p.exists() and (not z["verify_md5"] or md5sum(p) == md5)
        if ok:
            log.info("%s present%s", name, " (md5 ok)" if z["verify_md5"] else "")
            out[name] = p
            continue
        if not z["download"]:
            raise FileNotFoundError(f"{p} missing or md5 mismatch and zenodo.download=false")
        url = f"{BASE}/records/{z['record']}/files/{name}?download=1"
        tmp = p.with_suffix(p.suffix + ".part")
        log.info("downloading %s", url)
        t0 = time.time()
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done, last = 0, 0
            with open(tmp, "wb") as f:
                for b in r.iter_content(1 << 22):
                    f.write(b)
                    done += len(b)
                    if done - last > 256 << 20:
                        log.info("  %s: %.2f / %.2f GB", name, done / 1e9, total / 1e9)
                        last = done
        got = md5sum(tmp)
        if z["verify_md5"] and got != md5:
            raise RuntimeError(f"{name}: md5 {got} != expected {md5}; partial file kept at {tmp}")
        tmp.rename(p)
        log.info("  %s done in %.0fs (md5 ok)", name, time.time() - t0)
        out[name] = p
    return out
