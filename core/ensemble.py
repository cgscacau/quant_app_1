from __future__ import annotations
from typing import List, Dict




def weighted_ensemble(results: List[Dict], weights: Dict[str, float]) -> Dict:
present_models = {r['model'] for r in results}
w = {m: weights.get(m, 0.0) for m in present_models}
s = sum(w.values()) or 1.0
w = {m: v/s for m, v in w.items()}
p = sum(r['prob_up'] * w.get(r['model'], 0.0) for r in results)
return {'prob_up': p, 'weights': w}
