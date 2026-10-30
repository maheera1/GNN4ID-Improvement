import queue
import random
import threading
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

CLASS_NAMES = ['Benign', 'WebBased', 'Spoofing', 'Recon', 'Mirai', 'Dos', 'DDos', 'BruteForce']


class AsyncExplainer:
    """Runs explain_fn(payload) in background threads, so the detection loop never waits for it."""

    def __init__(self, explain_fn, num_workers=8, max_queue=1000):
        self.explain_fn = explain_fn
        self.q = queue.Queue(maxsize=max_queue)
        self.results = {}
        self.lock = threading.Lock()
        self.dropped = 0
        self.workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def submit(self, alert_id, payload):
        """Queue one explanation. If the queue is full it is skipped (the alert is still raised)."""
        with self.lock:
            self.results[alert_id] = {'t_submit': time.perf_counter(), 't_done': None, 'text': None}
        try:
            self.q.put_nowait((alert_id, payload))
        except queue.Full:
            with self.lock:
                self.results.pop(alert_id, None)
                self.dropped += 1
            return False
        return True

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            alert_id, payload = item
            try:
                text = self.explain_fn(payload)
            except Exception as e:          # a failing explanation must never kill the worker
                text = f"(explanation failed: {e!r})"
            with self.lock:
                self.results[alert_id]['text'] = text
                self.results[alert_id]['t_done'] = time.perf_counter()
            self.q.task_done()

    def wait_all(self):
        self.q.join()

    def stats(self):
        with self.lock:
            delays = [r['t_done'] - r['t_submit'] for r in self.results.values() if r['t_done']]
            return {'mean_delay_s': float(np.mean(delays)) if delays else float('nan'),
                    'max_delay_s': float(np.max(delays)) if delays else float('nan'),
                    'dropped': self.dropped}

    def shutdown(self):
        for _ in self.workers:
            self.q.put(None)
        for t in self.workers:
            t.join(timeout=2)


def make_simulated_llm(delay_scale=1.0, min_s=6.0, max_s=20.0, seed=0):
    """Stand-in for the real LLM: waits min_s..max_s seconds (times delay_scale), then returns text.
    The paper reports 6-20 s per explanation. delay_scale < 1 only shortens the DEMO."""
    rng = random.Random(seed)
    lock = threading.Lock()

    def explain(payload):
        with lock:
            delay = rng.uniform(min_s, max_s) * delay_scale
        time.sleep(delay)
        return (f"[SIMULATED LLM] Flow {payload['alert_id']} classified as "
                f"'{payload['predicted_class']}' ({payload['confidence'] * 100:.1f}% confidence).")
    return explain


def _load_graphs(test_dataset, n, seed, device):
    """n random test graphs, one per batch, already on the device (so only inference is timed)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_dataset), min(n, len(test_dataset)), replace=False).tolist()
    return [b.to(device) for b in DataLoader(Subset(test_dataset, idx), batch_size=1, shuffle=False)]


def _run_stream(model, graphs, device, mode, explain_fn, workers):
    """Feeds flows one at a time: detect -> (attack?) -> raise alert.
    mode 'blocking': the alert is raised only AFTER the explanation exists (old way).
    mode 'async'   : the alert is raised at once, the explanation is produced in the background."""
    use_cuda = str(device).startswith('cuda')
    explainer = AsyncExplainer(explain_fn, num_workers=workers) if mode == 'async' else None
    t_alert, t_detect = [], []
    t_start = time.perf_counter()
    with torch.no_grad():
        for i, g in enumerate(graphs):
            t0 = time.perf_counter()
            out = model(g.x_dict, g.edge_index_dict, g)
            conf, pred = out.exp().max(dim=1)
            pred, conf = int(pred), float(conf)          # int()/float() also wait for the GPU
            t_detect.append((time.perf_counter() - t0) * 1000.0)
            if pred != 0:
                payload = {'alert_id': i, 'predicted_class': CLASS_NAMES[pred], 'confidence': conf}
                if mode == 'blocking':
                    explain_fn(payload)                  # the alert has to wait for this
                else:
                    explainer.submit(i, payload)         # returns immediately
                t_alert.append((time.perf_counter() - t0) * 1000.0)
    stream_s = time.perf_counter() - t_start
    res = {'alerts': len(t_alert), 'mean_ms': float(np.mean(t_alert)), 'p95_ms': float(np.percentile(t_alert, 95)),
           'max_ms': float(np.max(t_alert)), 'detect_ms': float(np.mean(t_detect)),
           'stream_s': stream_s, 'total_s': stream_s, 'delay_s': float('nan'), 'dropped': 0}
    if explainer is not None:
        explainer.wait_all()
        st = explainer.stats()
        res.update({'total_s': time.perf_counter() - t_start, 'delay_s': st['mean_delay_s'], 'dropped': st['dropped']})
        explainer.shutdown()
    return res


def compare_pipelines(model, test_dataset, device, n_flows=200, delay_scale=0.01, workers=8, seed=0):
    """IMPROVEMENT 1 (async part): blocking vs asynchronous explanation, same flows, same simulated LLM."""
    model = model.to(device).eval()
    graphs = _load_graphs(test_dataset, n_flows, seed, device)
    for g in graphs[:5]:                                   # warm-up
        with torch.no_grad():
            model(g.x_dict, g.edge_index_dict, g)
    llm = make_simulated_llm(delay_scale=delay_scale)
    bl = _run_stream(model, graphs, device, 'blocking', llm, workers)
    llm = make_simulated_llm(delay_scale=delay_scale)
    asy = _run_stream(model, graphs, device, 'async', llm, workers)
    full_llm_ms = bl['detect_ms'] + 13000.0               # projection: detection + mean of 6-20 s

    rows = [
        {'Pipeline': f'Blocking (demo LLM, {delay_scale * 6:.2f}-{delay_scale * 20:.2f} s)', 'Alerts': bl['alerts'],
         'Mean time to alert (ms)': round(bl['mean_ms'], 1), '95th pct (ms)': round(bl['p95_ms'], 1),
         'Max (ms)': round(bl['max_ms'], 1), 'Mean explanation delay (s)': 'same as alert',
         'Total time (s)': round(bl['total_s'], 2)},
        {'Pipeline': 'Asynchronous (same demo LLM)', 'Alerts': asy['alerts'],
         'Mean time to alert (ms)': round(asy['mean_ms'], 2), '95th pct (ms)': round(asy['p95_ms'], 2),
         'Max (ms)': round(asy['max_ms'], 2), 'Mean explanation delay (s)': round(asy['delay_s'], 2),
         'Total time (s)': round(asy['total_s'], 2)},
        {'Pipeline': 'Blocking, full-size LLM 6-20 s (PROJECTION)', 'Alerts': '-',
         'Mean time to alert (ms)': round(full_llm_ms, 0), '95th pct (ms)': '-', 'Max (ms)': '-',
         'Mean explanation delay (s)': 'same as alert', 'Total time (s)': '-'},
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nDropped explanations (queue full): {asy['dropped']}")
    return df