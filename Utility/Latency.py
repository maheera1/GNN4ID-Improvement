import copy
import io
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from Utility.Training import test, measure_latency

# Newer PyTorch keeps quantization under torch.ao; older versions use torch.quantization
try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:
    from torch.quantization import quantize_dynamic


def _model_size_mb(model):
    """Size of the model weights in MB."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 * 1024)


def run_latency_benchmark(model, test_dataset, n_graphs=2000, max_batches=300, seed=42):
    """
    IMPROVEMENT 1: Benchmark GNN inference latency and apply INT8 dynamic quantization.

    Compares FP32 (GPU, CPU) against INT8-quantized (CPU) versions of the SAME trained
    model, at batch size 1 (real-time, flow by flow) and batch size 64 (batched).
    Reports latency per graph, throughput, accuracy, model size and speedup.

    Args:
        model: trained HeteroGNN (the model WITHOUT edge attributes).
        test_dataset: the test NIDSDataset.
        n_graphs: number of randomly chosen test graphs to benchmark on.

    Returns:
        pandas.DataFrame with the results table.
    """
    n_graphs = min(n_graphs, len(test_dataset))

    # Random subset (the test set is stored class by class, so a plain [:n] would be one class)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test_dataset), generator=g)[:n_graphs].tolist()
    subset = Subset(test_dataset, idx)
    loader_b1 = DataLoader(subset, batch_size=1, shuffle=False)
    loader_b64 = DataLoader(subset, batch_size=64, shuffle=False)

    # Build the three model variants from the same trained weights
    models = {}
    if torch.cuda.is_available():
        models['FP32 GPU'] = (copy.deepcopy(model).to('cuda').eval(), 'cuda')
    model_cpu = copy.deepcopy(model).to('cpu').eval()
    models['FP32 CPU'] = (model_cpu, 'cpu')
    # Dynamic INT8 quantization of all torch.nn.Linear layers (runs on CPU only)
    model_q = quantize_dynamic(copy.deepcopy(model_cpu), {nn.Linear}, dtype=torch.qint8)
    models['INT8 CPU'] = (model_q.eval(), 'cpu')

    rows = []
    for name, (m, dev) in models.items():
        print(f"Benchmarking {name} ...")
        acc = test(loader_b64, m, dev)
        size = _model_size_mb(m)
        for bs, loader in ((1, loader_b1), (64, loader_b64)):
            lat = measure_latency(m, loader, dev, max_batches=max_batches)
            rows.append({
                'Model': name,
                'Batch size': bs,
                'ms/graph': round(lat, 3),
                'Graphs/sec': round(1000.0 / lat, 1),
                'Accuracy': round(acc, 4),
                'Size (MB)': round(size, 2),
            })

    df = pd.DataFrame(rows)
    base = df[df['Model'] == 'FP32 CPU'].set_index('Batch size')['ms/graph']
    df['Speedup vs FP32 CPU'] = df.apply(
        lambda r: round(base[r['Batch size']] / r['ms/graph'], 2), axis=1)

    print("\n" + df.to_string(index=False))
    return df