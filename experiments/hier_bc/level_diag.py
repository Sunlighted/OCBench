import pickle, sys, collections
import numpy as np
for path in sys.argv[1:]:
    eps = pickle.load(open(path, 'rb'))
    print(f'== {path.split("/")[-2]}  success {np.mean([e["success"] for e in eps]):.3f}  n={len(eps)}')
    lv = collections.defaultdict(collections.Counter); steps = collections.defaultdict(list)
    support = collections.Counter(); held = collections.Counter()
    for e in eps:
        for ev in e['events']:
            z = round(float(ev['target_pos'][2]), 2)
            lv[z][ev['outcome']] += 1
            if ev['outcome'] == 'done':
                steps[z].append(ev['steps'])
            if ev['outcome'] == 'timeout' and 'diag' in ev:
                d = ev['diag']; tp = ev['target_pos']
                blocks = [d[k] for k in d if k.startswith('privileged/block_') and k.endswith('_pos')]
                held[z] += float(d['proprio/gripper_contact'][0]) > 0.5
                if tp[2] > 0.05:
                    ok = any(np.linalg.norm(b[:2] - tp[:2]) < 0.02 and abs(b[2] - (tp[2] - 0.06)) < 0.02 for b in blocks)
                    support[(z, 'support_ok' if ok else 'support_missing')] += 1
    for z in sorted(lv):
        c = lv[z]; tot = c['done'] + c['timeout']
        print(f'  z={z:.2f}: done {c["done"]:4d} timeout {c["timeout"]:4d} ep_end {c["episode_end"]:4d}'
              f'  done_rate {c["done"] / max(tot, 1):.2f}  med_steps {np.median(steps[z]) if steps[z] else -1:.0f}'
              f'  timeouts_holding {held[z]}')
    print('  timeout support:', dict(support))
    print('  done per ep:', sorted(collections.Counter(sum(ev['outcome'] == 'done' for ev in e['events']) for e in eps).items()))
