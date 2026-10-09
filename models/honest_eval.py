"""
Split-coverage audit
====================
Which injected failure families does each time split contain? The temporal
test window (days 153-179) holds only gradual drift, so a metric computed there
cannot support claims about cascades, silent failures or phantom recoveries.
`python models/train.py` therefore also runs the leave-one-family-out protocol.
"""
import pandas as pd

from models.evaluation import EVENT_MODES
from models.train import temporal_split


def split_failure_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in zip(["train", "val", "test"], temporal_split(df)):
        counts = part["event_family"].value_counts()
        rec = {"split": name, "first_day": int(part["day_idx"].min()), "last_day": int(part["day_idx"].max())}
        rec.update({m: int(counts.get(m, 0)) for m in EVENT_MODES})
        rows.append(rec)
    return pd.DataFrame(rows)


def uncovered_failure_modes(df: pd.DataFrame, split: str = "test") -> set:
    cov = split_failure_coverage(df).set_index("split").loc[split]
    return {m for m in EVENT_MODES if cov[m] == 0}


if __name__ == "__main__":
    from models.train import load_feature_matrix
    frame = load_feature_matrix()
    print(split_failure_coverage(frame).to_string(index=False))
    print("absent from test:", sorted(uncovered_failure_modes(frame)))
