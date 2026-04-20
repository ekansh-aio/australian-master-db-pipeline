import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime


def _sanitize(obj):
    """
    Recursively sanitize data for JSON serialization.
    - NaN / Inf floats → 0.0
    - numpy scalars → Python native types
    - numpy arrays → lists
    - Path / datetime → strings
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return 0.0
        return val
    elif isinstance(obj, float) and not math.isfinite(obj):
        return 0.0
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.generic):
            val = o.item()
            if isinstance(val, float) and not math.isfinite(val):
                return 0.0
            return val
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def safe_json_dump(data, file_path, indent=2):
    """Dump pipeline data safely to a JSON file. Handles numpy, NaN, Inf, datetime, Path."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(data), f, indent=indent, ensure_ascii=False)


def safe_json_dumps(data, indent=2):
    """Serialize pipeline data to a JSON string. Handles numpy, NaN, Inf, datetime, Path."""
    return json.dumps(_sanitize(data), indent=indent, ensure_ascii=False)
