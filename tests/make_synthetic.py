"""Synthetic CASAS-like zips with several line layouts and defects (for testing the inventory only)."""
import gzip
import io
import random
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
SENS = ["Bedroom", "BedroomArea", "Bed", "Bathroom", "BathroomArea", "Kitchen", "KitchenArea", "LivingRoom",
        "LivingRoomArea", "OfficeChair", "DiningRoom", "Hall", "FrontDoor", "BackDoor"]


def events(days, start="2012-01-01", sensors=SENS, gap=None, back=0):
    t = datetime.fromisoformat(start)
    end = t + timedelta(days=days)
    rows = []
    while t < end:
        h = t.hour
        rate = 40 if 7 <= h < 23 else 4
        t += timedelta(seconds=rng.expovariate(rate / 3600))
        if gap and gap[0] <= t < gap[1]:
            t = gap[1]
        s = rng.choice(sensors)
        msg = rng.choice(["OPEN", "CLOSE"]) if s.endswith("Door") else rng.choice(["ON", "OFF"])
        rows.append((t, s, msg))
    for _ in range(back):                       # a few timestamps out of order
        i = rng.randrange(1, len(rows))
        rows[i] = (rows[i][0] - timedelta(seconds=30), rows[i][1], rows[i][2])
    return rows


def fmt(rows, lay, label=False):
    L = []
    for t, s, m in rows:
        ts = t.strftime("%Y-%m-%d %H:%M:%S.%f")
        lab = ["Sleep", "begin"] if (label and rng.random() < 0.01) else []
        if lay == "space":
            L.append(" ".join([ts, s, m] + lab))
        elif lay == "tab":
            L.append("\t".join([ts, s, m] + ([" ".join(lab)] if lab else [])))
        else:
            L.append(",".join([ts.replace(" ", "T"), s, m] + ([" ".join(lab)] if lab else [])))
    return "\n".join(L) + "\n"


h101 = events(80, gap=(datetime(2012, 1, 20), datetime(2012, 1, 23)), back=5)
txt101 = fmt(h101, "space")
lines = txt101.split("\n")
lines.insert(100, "garbage line without timestamp")
lines.insert(200, "2012-01-02 25:99:00 Bedroom ON")
txt101 = "\n".join(lines)
h102 = events(60, start="2013-03-01", sensors=SENS[:8])
h103 = events(10, start="2014-05-01")
tm001 = events(70, start="2015-02-01")

with zipfile.ZipFile(out / "data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("data/hh101.txt", txt101)
    z.writestr("data/hh102.txt", "Date\tTime\tSensor\tMessage\n" + fmt(h102, "tab"))
    z.writestr("data/hh103.txt.gz", gzip.compress(fmt(h103, "space").encode()))
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zi:
        zi.writestr("tm001.csv", fmt(tm001, "comma"))
    z.writestr("data/tm.zip", inner.getvalue())
    z.writestr("__MACOSX/data/._hh101.txt", "junk")
    z.writestr("data/README.txt", "readme")
with zipfile.ZipFile(out / "labeled_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("labeled/hh101.txt", fmt(h101[:20000], "space", label=True))
with zipfile.ZipFile(out / "floorplans.zip", "w") as z:
    z.writestr("floorplans/hh101.png", b"png")
    z.writestr("floorplans/hh102.jpg", b"jpg")
print("ok", out)
