"""Format-tolerant streaming parser for CASAS event lines.

Accepted layouts (sniffed per line, so mixed files still parse):
  2011-06-15 03:38:09.546797 Bedroom ON                      (whitespace)
  2011-06-15 03:38:09.546797\tBedroom\tON\t<extra...>         (tab)
  2011-06-15T03:38:09.546797,Bedroom,ON,<extra...>            (comma, ISO 'T')
Anything after the message is kept as `extra` (activity labels in the labelled files).
"""
import re
from datetime import date

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$")
DT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)$")

_ord_cache = {}


def day_seconds(d):
    v = _ord_cache.get(d)
    if v is None:
        v = date(int(d[0:4]), int(d[5:7]), int(d[8:10])).toordinal() * 86400
        _ord_cache[d] = v
    return v


def tod_seconds(t):
    h, m, s = t.split(":")
    h, m, s = int(h), int(m), float(s)
    if h > 23 or m > 59 or s >= 61:
        raise ValueError("time out of range")
    return h * 3600 + m * 60 + s


def split(line):
    if "\t" in line:
        return [p.strip() for p in line.split("\t")], "tab"
    if "," in line and " " not in line.split(",")[0][:10]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            return parts, "comma"
    return line.split(), "space"


def parse(line):
    """-> (day 'YYYY-MM-DD', epoch seconds float, sensor, message, extra list, layout) or None."""
    parts, lay = split(line)
    if len(parts) < 3:
        return None
    m = DT.match(parts[0])
    if m:
        d, t, rest = m.group(1), m.group(2), parts[1:]
    elif len(parts) >= 4 and DATE.match(parts[0]) and TIME.match(parts[1]):
        d, t, rest = parts[0], parts[1], parts[2:]
    else:
        return None
    if len(rest) < 2:
        return None
    try:
        ts = day_seconds(d) + tod_seconds(t)
    except ValueError:
        return None
    return d, ts, rest[0], rest[1], rest[2:], lay
