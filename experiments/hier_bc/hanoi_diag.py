import pickle, sys, collections
import numpy as np
eps = pickle.load(open(sys.argv[1], 'rb'))
print('succ', np.mean([e['success'] for e in eps]), 'n', len(eps))
d0 = next(ev['diag'] for e in eps for ev in e['events'] if 'diag' in ev)
print('keys', sorted(d0.keys()))
cats = collections.Counter()
rows = []
for e in eps:
    if e['success']:
        continue
    ev = e['events'][-1]
    if 'diag' not in ev:
        cats['no_diag'] += 1
        continue
    d = ev['diag']; t = d['target']
    disk = int(t['disk']); dp = d[f'privileged/disk_{disk}_pos']; gp = np.asarray(t['disk_goal_pos'])
    xy = np.linalg.norm(dp[:2] - gp[:2]); dz = dp[2] - gp[2]
    contact = float(d['proprio/gripper_contact'][0])
    ee = d['proprio/effector_pos']; ee_d = np.linalg.norm(ee - dp)
    # which disk is closest to the effector
    if d['failure']:
        c = 'failure(terminated)'
    elif xy <= 0.04 and dz > 0.03:
        c = 'at_peg_xy_but_too_high'
    elif xy <= 0.04 and dz < -0.03:
        c = 'at_peg_xy_but_too_low'
    elif contact > 0.5:
        c = 'holding_but_far_from_target'
    else:
        c = 'not_holding_far_from_target'
    done = sum(x['outcome'] == 'done' for x in e['events'])
    cats[c] += 1
    rows.append((c, done, e['episode_length'], disk, round(xy, 3), round(dz, 3), round(contact, 2), round(ee_d, 3)))
print(cats)
for r in sorted(rows)[:: max(1, len(rows) // 30)]:
    print(r)
