import re
import pandas as pd

ROOMS = ["Bathroom", "Bedroom", "Kitchen", "LivingRoom", "DiningRoom", "Office", "Hallway", "Entryway", "Hall", "Garage",
         "LaundryRoom", "Laundry", "Closet", "Den", "Basement", "Stairway", "Porch", "Patio", "Foyer", "Pantry", "Study",
         "SewingRoom", "Workshop", "OtherRoom", "Lounge", "FamilyRoom", "Bath"]
ROOM_RE = re.compile(r"^(" + "|".join(sorted(ROOMS, key=len, reverse=True)) + r")([A-Z](?=[A-Z0-9]|$))?(.*)$", re.I)
PERIM = re.compile(r"^(Main|Outside|Front|Back)?(Entryway|Door|Entry|Window)", re.I)


def parse_name(n):
    """sensor name -> (room_type, room_id, spot). room_id = type + letter (one physical room)."""
    m = ROOM_RE.match(n)
    if m:
        rt = next(r for r in ROOMS if r.lower() == m.group(1).lower())
        if rt == "Bath":
            rt = "Bathroom"
        letter = (m.group(2) or "").upper()
        spot = m.group(3) or "Room"
        # 'LaundryRoomA...' style: Laundry + 'RoomA' -> room LaundryRoom
        mm = re.match(r"^Room([A-Z])?(.*)$", spot)
        if rt in ("Laundry",) and mm:
            rt, letter, spot = "LaundryRoom", (mm.group(1) or ""), (mm.group(2) or "Room")
        return rt, rt + letter, spot
    if PERIM.match(n):
        return "Perimeter", "Perimeter", n
    return "Other", "Other", n


def expand_homes(cell):
    out = []
    for part in re.split(r"[,\s]+(?=[a-z])", str(cell).strip()):
        part = part.strip().strip(",")
        if not part:
            continue
        m = re.fullmatch(r"([a-z]+)(\d+)-([a-z]+)?(\d+)", part)
        if m:
            pre, a, b = m.group(1), m.group(2), m.group(4)
            for i in range(int(a), int(b) + 1):
                out.append(f"{pre}{str(i).zfill(len(a))}")
        else:
            out.append(part)
    return out


def residents(table_csv):
    t = pd.read_csv(table_csv, header=None, skiprows=1, dtype=str).fillna("")
    res = {}
    for _, r in t.iterrows():
        for hc, nc in ((0, 1), (3, 4), (6, 7)):
            if hc + 1 >= len(r) or r[hc].strip().lower().startswith("home"):
                continue
            if r[hc].strip() and r[nc].strip():
                for h in expand_homes(r[hc]):
                    res[h] = r[nc].strip()
    return res
