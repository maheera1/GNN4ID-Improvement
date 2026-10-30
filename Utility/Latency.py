import copy
import io
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear as PygLinear

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


def _swap_pyg_linear(module):
    """PyTorch can only quantize torch.nn.Linear. The SAGE/GAT layers use PyG's own Linear class,
    so we first replace each one by a normal nn.Linear with the same weights."""
    swapped = 0
    for name, child in module.named_children():
        if isinstance(child, PygLinear):
            if isinstance(child.weight, nn.parameter.UninitializedParameter):
                raise RuntimeError("Model still has uninitialised (lazy) layers - run one batch through it first.")
            new = nn.Linear(child.in_channels, child.out_channels, bias=child.bias is not None)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
            swapped += 1
        else:
            swapped += _swap_pyg_linear(child)
    return swapped


def quantize_gnn_int8(model, swap_convs=True):
    """
    Returns (int8_model, n_swapped_layers). The model is copied, moved to CPU and quantized.
    swap_convs=False quantizes only the classifier head (the nn.Linear layers that already exist).
    swap_convs=True  also converts the graph-convolution layers, then quantizes everything.
    """
    m = copy.deepcopy(model).to('cpu').eval()
    n = _swap_pyg_linear(m) if swap_convs else 0
    q = quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
    return q.eval(), n


def run_latency_benchmark(model, test_dataset, n_graphs=2000, max_batches=300, seed=42):
    """
    IMPROVEMENT 1: Benchmark GNN inference latency and INT8 dynamic quantization.

    Variants of the SAME trained model:
      FP32 GPU, FP32 CPU, INT8 CPU (head only), INT8 CPU (convs + head)
    at batch size 1 (real-time) and 64 (batched).
    """
    n_graphs = min(n_graphs, len(test_dataset))

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test_dataset), generator=g)[:n_graphs].tolist()
    subset = Subset(test_dataset, idx)
    loader_b1 = DataLoader(subset, batch_size=1, shuffle=False)
    loader_b64 = DataLoader(subset, batch_size=64, shuffle=False)

    models = {}
    if torch.cuda.is_available():
        models['FP32 GPU'] = (copy.deepcopy(model).to('cuda').eval(), 'cuda')
    model_cpu = copy.deepcopy(model).to('cpu').eval()
    models['FP32 CPU'] = (model_cpu, 'cpu')

    # Sanity check: swapping PyG Linear -> nn.Linear must NOT change the predictions
    swapped_fp32 = copy.deepcopy(model_cpu)
    n_sw = _swap_pyg_linear(swapped_fp32)
    acc_orig = test(loader_b64, model_cpu, 'cpu')
    acc_swap = test(loader_b64, swapped_fp32.eval(), 'cpu')
    print(f"Swapped {n_sw} PyG Linear layers. FP32 accuracy before/after swap: "
          f"{acc_orig:.4f} / {acc_swap:.4f} (should be identical)")

    q_head, _ = quantize_gnn_int8(model, swap_convs=False)
    q_full, _ = quantize_gnn_int8(model, swap_convs=True)
    models['INT8 CPU (head only)'] = (q_head, 'cpu')
    models['INT8 CPU (convs+head)'] = (q_full, 'cpu')

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