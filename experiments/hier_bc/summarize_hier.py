"""Summarize hier_bc runs: python summarize_hier.py <dir containing <env>/eval.csv and final_eval_1000000.json>"""

import csv
import glob
import json
import os
import sys

root = sys.argv[1]
for run in sorted(glob.glob(os.path.join(root, '*-v0'))):
    name = os.path.basename(run)
    curve = []
    if os.path.exists(os.path.join(run, 'eval.csv')):
        for r in csv.DictReader(open(os.path.join(run, 'eval.csv'))):
            curve.append(f"{int(r['step']) // 1000}k:{100 * float(r['evaluation/success']):.0f}")
    print(f'=== {name}  curve(100 eps): {" ".join(curve)}')
    path = os.path.join(run, 'final_eval_1000000.json')
    if not os.path.exists(path):
        print('  (final eval not finished)')
        continue
    res = json.load(open(path))
    for k, m in res.items():
        tasks = ' '.join(
            f"{key.split('/')[1]}={100 * v:.0f}%/{m.get(key.replace('done_rate', 'done_steps_median'), -1):.0f}"
            for key, v in m.items()
            if key.endswith('/done_rate')
        )
        print(
            f"  {k} T={m['max_episode_steps']}: succ {100 * m['success']:.1f}±{100 * m['success_se']:.1f}"
            f"  succ_len_med {m['success_length_median']:.0f}  done/ep {m['subgoals_done_per_ep']:.2f}"
            f"  tmo/ep {m['timeouts_per_ep']:.2f}  tmo_success_now {m.get('timeout_success_now', 0):.2f}"
            f"  done_steps med/p90 {m.get('done_steps_median', -1):.0f}/{m.get('done_steps_p90', -1):.0f}"
            f"  [subtask done%/median steps: {tasks}]"
        )
