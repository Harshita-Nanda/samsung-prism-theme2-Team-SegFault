"""
Prints RAW BM25 / cosine scores from the REAL all-MiniLM-L6-v2 catalog for
positive, unrelated and gibberish queries, in exactly the query format
mapper.py builds ("<actionName>. <description>. <steps>"), so the
sentence-transformers floors in mapper.EVIDENCE_THRESHOLDS_BY_MODE can be set
from data instead of guessed.

Run (needs network the first time to fetch the model, then cached):
    python calibrate_thresholds.py

Decision rule: pick min_vector_score between the highest 'NEGATIVE' top cosine
and the lowest 'POSITIVE' top cosine; likewise for BM25. If the ranges overlap,
the floors alone cannot separate them -- raise that with the team.
"""
from catalog import DeeplinkCatalog
from mapper import EVIDENCE_THRESHOLDS_BY_MODE

POSITIVE = {
    "backup (expect DL-0542)": ("Back Up Phone Data", "It will let you back up your files",
        ["Navigate to and open Settings.", "Tap on Accounts and backup.", "Select Back up data to secure your personal files."]),
    "wifi": ("Enable Wi-Fi", "It will connect you to a network", ["Open Settings, tap Connections, tap Wi-Fi and turn it on."]),
    "bluetooth": ("Turn on Bluetooth", "It will let you connect devices", ["Tap Bluetooth and enable it."]),
    "auto sync": ("Disable auto sync", "It will stop background syncing", ["Tap Accounts and backup and turn off auto sync of personal account data."]),
    "nav bar": ("Configure Navigation Bar Settings", "It will let you choose navigation type",
        ["Tap Display.", "Tap Navigation bar.", "Select Swipe gestures."]),
}
NEGATIVE = {
    "gibberish": ("Xqzv Plmk", "Wrrt zzkq", ["Qwxz vbnm jjkl."]),
    "unrelated (banana)": ("Quarterly Banana Stock Report", "It will list the banana prices", ["Read the quarterly banana report."]),
    "unrelated (cooking)": ("Bake Sourdough Bread", "It will help you bake bread", ["Mix flour and water.", "Let the dough rise overnight."]),
    "generic nav only": ("Open Settings", "It will open Settings", ["Navigate to and open Settings."]),
    "hardware": ("Replace Cracked Glass", "It will get the screen repaired", ["Take the device to a repair technician."]),
}

catalog = DeeplinkCatalog("deeplinks.json")  # real model; raises if unavailable
t = EVIDENCE_THRESHOLDS_BY_MODE[catalog.retrieval_mode]
print(f"mode={catalog.retrieval_mode}  current floors: bm25>={t.min_bm25_score}  cosine>={t.min_vector_score}\n")

tops = {"POSITIVE": [], "NEGATIVE": []}
for group, queries in (("POSITIVE", POSITIVE), ("NEGATIVE", NEGATIVE)):
    for label, (name, desc, steps) in queries.items():
        q = f"{name}. {desc}. {' '.join(steps)}"
        ms = catalog.search(q, top_k=3)
        best_cos = max(m.vector_score for m in ms)
        best_bm = max(m.bm25_score for m in ms)
        tops[group].append((best_bm, best_cos))
        print(f"[{group}] {label:26s} top ids={[m.entry.id for m in ms]}  max bm25={best_bm:6.2f}  max cos={best_cos:.3f}")

print("\nPOSITIVE  min bm25 = %.2f   min cos = %.3f" % (min(b for b, _ in tops["POSITIVE"]), min(c for _, c in tops["POSITIVE"])))
print("NEGATIVE  max bm25 = %.2f   max cos = %.3f" % (max(b for b, _ in tops["NEGATIVE"]), max(c for _, c in tops["NEGATIVE"])))
