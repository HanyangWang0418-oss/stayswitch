"""Deterministic verifier: compares /app/answer.txt to tests/expected.json. No LLM, no network."""
import json, math, re, sys
from pathlib import Path

def norm_str(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def parse_number(s):
    s = s.strip().replace(",", "").replace("$", "").replace("−", "-")
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\)?", s)
    if not m:
        raise ValueError(s)
    v = float(m.group(1))
    return -v if s.startswith("(") else v

def check(spec, got):
    t = spec["type"]
    if t == "number":
        try:
            g = parse_number(got)
        except ValueError:
            return False
        e = float(spec["value"])
        return math.isclose(g, e, rel_tol=spec.get("rel_tol", 0.0), abs_tol=spec.get("abs_tol", 0.0))
    if t == "string":
        return norm_str(got) == norm_str(str(spec["value"]))
    if t == "list":  # order-insensitive set of strings or numbers, comma/newline separated
        items = [x for x in re.split(r"[,\n;]", got) if x.strip()]
        if spec.get("numeric"):
            try:
                g = sorted(parse_number(x) for x in items)
            except ValueError:
                return False
            e = sorted(float(x) for x in spec["value"])
            return len(g) == len(e) and all(math.isclose(a, b, rel_tol=spec.get("rel_tol", 0.0), abs_tol=spec.get("abs_tol", 0.0)) for a, b in zip(g, e))
        return sorted(norm_str(x) for x in items) == sorted(norm_str(str(x)) for x in spec["value"])
    raise ValueError(t)

def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    ans = Path(sys.argv[2])
    ok = False
    if ans.is_file():
        lines = [l for l in ans.read_text().strip().splitlines() if l.strip()]
        got = "\n".join(lines) if spec["type"] == "list" else (lines[-1] if lines else "")
        ok = check(spec, got)
    out = Path(sys.argv[3]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("1" if ok else "0")
    print("reward", int(ok))

if __name__ == "__main__":
    main()
