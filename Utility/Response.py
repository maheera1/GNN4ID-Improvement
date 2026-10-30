import copy
import time
from collections import deque

import numpy as np
import pandas as pd

CLASS_NAMES = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']
ORDER = {'allow': 0, 'alert': 1, 'rate_limit': 2, 'block': 3}

DEFAULT_POLICY = {
    'name': 'balanced (our policy)',
    'min_confidence': 0.60,                  # attack detections below this go to a human analyst, no automatic action
    'window_s': 60.0,                        # repeated detections from the same host are counted inside this window
    'ttl_s': {'rate_limit': 120.0, 'block': 300.0},   # every rule expires automatically
    'ratelimit_drop': 0.8,                   # a rate-limited host loses 80% of its traffic
    'max_blocks_per_min': 20,                # circuit breaker: at most this many NEW blocks per minute
    'allowlist': ['10.0.0.1', '10.0.0.2', '10.0.0.3'],   # protected servers: never limited or blocked
    'severity': {'Recon': 'low', 'WebBased': 'medium', 'BruteForce': 'high', 'Spoofing': 'high',
                 'Dos': 'high', 'DDos': 'critical', 'Mirai': 'critical'},
    # ladder[severity] = [(evidence_needed, action), ...]. Evidence = number of attack detections from the same
    # host inside the window. The last row whose evidence_needed has been reached decides the action.
    'ladder': {
        'low':      [(1, 'alert'), (8, 'rate_limit')],
        'medium':   [(1, 'alert'), (3, 'rate_limit'), (8, 'block')],
        'high':     [(1, 'alert'), (2, 'rate_limit'), (4, 'block')],
        'critical': [(1, 'alert'), (2, 'block')],
    },
}


def policy_suite(base=None):
    """Four policies compared on the same traffic: today's paper, cautious, ours, trigger-happy."""
    base = copy.deepcopy(base or DEFAULT_POLICY)
    sevs = ('low', 'medium', 'high', 'critical')

    detect = copy.deepcopy(base)
    detect['name'] = 'detect-only (the paper today)'
    detect['ladder'] = {s: [(1, 'alert')] for s in sevs}

    cons = copy.deepcopy(base)
    cons['name'] = 'conservative'
    cons['min_confidence'] = 0.85
    cons['ladder'] = {'low': [(1, 'alert')],
                      'medium': [(1, 'alert'), (6, 'rate_limit')],
                      'high': [(1, 'alert'), (4, 'rate_limit'), (8, 'block')],
                      'critical': [(1, 'alert'), (4, 'block')]}

    aggr = copy.deepcopy(base)
    aggr['name'] = 'aggressive (block on first alert)'
    aggr['min_confidence'] = 0.5
    aggr['allowlist'] = []
    aggr['max_blocks_per_min'] = 10 ** 9
    aggr['ladder'] = {s: [(1, 'block')] for s in sevs}
    return [detect, cons, base, aggr]


class SimulatedEnforcer:
    """DRY RUN: only returns the firewall command it WOULD run. It never executes anything."""
    @staticmethod
    def command_for(host, action):
        if action == 'block':
            return f"iptables -I INPUT -s {host} -j DROP"
        if action == 'rate_limit':
            return f"iptables -I INPUT -s {host} -m limit --limit 20/second -j ACCEPT"
        return ''


def build_stream(y_true, y_pred, conf, n_attack=3000, benign_share=0.9, seed=0, rate=50.0,
                 n_benign_hosts=300, attackers_per_class=4):
    """
    Turns the model's per-flow outputs into a time-ordered event stream with SIMULATED source hosts.
    (The project drops real IP addresses from the data, so hosts are made up: 3 protected servers that carry
    10x more traffic, n_benign_hosts normal hosts, and a few attacker hosts per attack class.)
    benign_share=0.9 means 9 of 10 flows are benign, like on a real network. rate = flows per second.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred, conf = np.asarray(y_true), np.asarray(y_pred), np.asarray(conf)
    att, ben = np.where(y_true != 0)[0], np.where(y_true == 0)[0]
    n_att = min(n_attack, len(att))
    n_ben = int(round(n_att * benign_share / (1.0 - benign_share)))
    att_sel = rng.choice(att, n_att, replace=False)
    ben_sel = rng.choice(ben, n_ben, replace=n_ben > len(ben))     # benign flows are re-used if needed

    ben_hosts = ['10.0.0.1', '10.0.0.2', '10.0.0.3'] + [f"10.1.{i // 250}.{i % 250 + 1}" for i in range(n_benign_hosts)]
    w = np.array([10.0] * 3 + [1.0] * n_benign_hosts)
    w = w / w.sum()
    host_kind = {h: 'benign' for h in ben_hosts}

    events = []
    for idx, hi in zip(ben_sel, rng.choice(len(ben_hosts), size=n_ben, p=w)):
        events.append({'host': ben_hosts[int(hi)], 'true': 0, 'idx': int(idx)})
    for idx, k in zip(att_sel, rng.integers(1, attackers_per_class + 1, size=n_att)):
        h = f"172.16.{int(y_true[idx])}.{int(k)}"
        host_kind[h] = 'attacker'
        events.append({'host': h, 'true': int(y_true[idx]), 'idx': int(idx)})

    stream = []
    for k, j in enumerate(rng.permutation(len(events))):
        e = events[int(j)]
        e['t'] = k / rate
        e['pred'] = int(y_pred[e['idx']])
        e['conf'] = float(conf[e['idx']])
        stream.append(e)
    return stream, host_kind


class ResponseEngine:
    """Applies one policy to the event stream. Safety guards: low-confidence detections are not acted on,
    evidence must repeat, protected hosts are never blocked, a circuit breaker limits blocks per minute,
    every rule expires (TTL), and every enforcement is written to an audit log."""

    def __init__(self, policy, class_names=CLASS_NAMES, audit_cap=5000):
        self.p = policy
        self.names = class_names
        self.enforcer = SimulatedEnforcer()
        self.audit_cap = audit_cap

    def run(self, stream, host_kind):
        p = self.p
        allow = set(p['allowlist'])
        active, evid, recent_blocks, audit = {}, {}, deque(), []
        counts = {'alert': 0, 'rate_limit': 0, 'block': 0}
        n_att = n_ben = 0
        stop_att = hard_att = stop_ben = 0.0
        n_low = n_prot = n_trip = 0
        first_seen, leaked, leak_at_block, sec_to_block = {}, {}, {}, {}
        ever_blocked, seen_benign = set(), set()
        t_dec = 0.0

        for e in stream:
            t, h, is_att = e['t'], e['host'], e['true'] != 0
            c0 = time.perf_counter()
            if is_att:
                n_att += 1
                first_seen.setdefault(h, t)
            else:
                n_ben += 1
                seen_benign.add(h)

            a = active.get(h)
            if a and a[1] <= t:                 # rule expired
                del active[h]
                a = None
            cur = a[0] if a else 'allow'

            if cur == 'block':                  # the firewall drops it; the detector never sees it
                if is_att:
                    stop_att += 1
                    hard_att += 1
                else:
                    stop_ben += 1
                t_dec += time.perf_counter() - c0
                continue
            if cur == 'rate_limit':
                if is_att:
                    stop_att += p['ratelimit_drop']
                else:
                    stop_ben += p['ratelimit_drop']
            if is_att and h not in leak_at_block:
                leaked[h] = leaked.get(h, 0) + 1

            if e['pred'] != 0:
                cls = self.names[e['pred']]
                if e['conf'] < p['min_confidence']:
                    n_low += 1                  # not sure enough: a human analyst looks at it
                else:
                    dq = evid.setdefault(h, deque())
                    dq.append(t)
                    while dq and dq[0] < t - p['window_s']:
                        dq.popleft()
                    n = len(dq)
                    sev = p['severity'].get(cls, 'low')
                    target = 'alert'
                    for need, act in p['ladder'][sev]:
                        if n >= need:
                            target = act
                    reason = 'policy'
                    if target != 'alert' and h in allow:
                        target, reason = 'alert', 'protected host'
                        n_prot += 1
                    if target == 'block':
                        while recent_blocks and recent_blocks[0] <= t - 60.0:
                            recent_blocks.popleft()
                        if len(recent_blocks) >= p['max_blocks_per_min']:
                            target, reason = 'rate_limit', 'circuit breaker'
                            n_trip += 1
                    if target == 'alert':
                        counts['alert'] += 1
                    elif ORDER[target] > ORDER[cur]:
                        ttl = p['ttl_s'][target]
                        active[h] = (target, t + ttl)
                        counts[target] += 1
                        if target == 'block':
                            recent_blocks.append(t)
                            ever_blocked.add(h)
                            if host_kind.get(h) == 'attacker' and h not in leak_at_block:
                                leak_at_block[h] = leaked.get(h, 0)
                                sec_to_block[h] = t - first_seen.get(h, t)
                        if len(audit) < self.audit_cap:
                            audit.append({'time_s': round(t, 1), 'host': h, 'detected_as': cls,
                                          'true_class': self.names[e['true']], 'confidence': round(e['conf'], 3),
                                          'evidence': n, 'action': target, 'reason': reason, 'ttl_s': ttl,
                                          'command (dry run)': self.enforcer.command_for(h, target)})
            t_dec += time.perf_counter() - c0

        attackers = {h for h, k in host_kind.items() if k == 'attacker' and h in first_seen}
        contained = attackers & ever_blocked
        leaks = [leak_at_block[h] for h in contained]
        secs = [sec_to_block[h] for h in contained]
        metrics = {
            'policy': p['name'],
            'attack_stopped_pct': 100.0 * stop_att / max(n_att, 1),
            'benign_wrongly_stopped_pct': 100.0 * stop_ben / max(n_ben, 1),
            'attackers_contained': f"{len(contained)}/{len(attackers)}",
            'benign_hosts_blocked': f"{len(seen_benign & ever_blocked)}/{len(seen_benign)}",
            'median_attack_flows_before_block': float(np.median(leaks)) if leaks else float('nan'),
            'median_seconds_to_block': float(np.median(secs)) if secs else float('nan'),
            'alerts': counts['alert'], 'rate_limits': counts['rate_limit'], 'blocks': counts['block'],
            'sent_to_analyst': n_low, 'protected_host_suppressions': n_prot, 'breaker_trips': n_trip,
            'decision_us': 1e6 * t_dec / max(len(stream), 1),
        }
        return metrics, audit


def compare_policies(stream, host_kind, base_policy=None):
    """Runs all four policies on the SAME stream. Returns (results table, list of metric dicts, audit logs)."""
    results, audits = [], {}
    for pol in policy_suite(base_policy):
        m, audit = ResponseEngine(pol).run(stream, host_kind)
        results.append(m)
        audits[pol['name']] = audit
    df = pd.DataFrame(results).round(2)
    df.columns = ['Policy', 'Attack traffic stopped %', 'Benign traffic wrongly stopped %', 'Attackers contained',
                  'Benign hosts blocked', 'Median attack flows before block', 'Median seconds to block',
                  'Alerts', 'Rate-limits', 'Blocks', 'Sent to analyst', 'Protected-host suppressions',
                  'Circuit-breaker trips', 'Decision time (us)']
    return df, results, audits


def plot_policy_comparison(results, path):
    import matplotlib
    import matplotlib.pyplot as plt
    names = [r['policy'] for r in results]
    ys = np.arange(len(names))
    h = 0.35
    fig, ax = plt.subplots(figsize=(9, 1.0 * len(names) + 1.6))
    for k, (key, lab, col) in enumerate((('attack_stopped_pct', 'Attack traffic stopped (higher is better)', '#2a78d6'),
                                         ('benign_wrongly_stopped_pct', 'Benign traffic wrongly stopped (lower is better)', '#eb6834'))):
        vals = [r[key] for r in results]
        bars = ax.barh(ys + (k - 0.5) * h, vals, height=h * 0.92, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(min(v + 1, 101), b.get_y() + b.get_height() / 2, f"{v:.1f}%", va='center', fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel('% of that kind of traffic')
    ax.set_title('What each response policy does to attack and benign traffic')
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()