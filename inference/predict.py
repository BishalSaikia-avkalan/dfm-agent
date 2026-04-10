import os
import torch
import numpy as np

from models.diffusionnet import DFM_DiffusionNet
from models.node_diffusionnet import NodeLevel_DFM_DiffusionNet
from models.gat import DFM_GAT
from models.pointnet import DFM_PointNetPP
from models.meshcnn import DFM_MeshCNN
from preprocessing.mesh_to_graph import process_stl_for_inference, CLF_NAMES

# Default paths to trained weights
_WEIGHTS_DIR = os.path.dirname(os.path.dirname(__file__))

_WEIGHTS_PATHS = {
    'DiffusionNet': os.path.join(_WEIGHTS_DIR, "DiffusionNet", "diffusionnet_dfm_best_5thApril.pt"),
    'NodeLevel_DiffusionNet': os.path.join(_WEIGHTS_DIR, "Node_level_DiffusionNet", "nodelevel_diffusionnet_best.pt"),
    'GAT': os.path.join(_WEIGHTS_DIR, "GAT", "gat_dfm_best.pt"),
    'PointNet++': os.path.join(_WEIGHTS_DIR, "PointNet++", "pointnetpp_dfm_best.pt"),
    'MeshCNN': os.path.join(_WEIGHTS_DIR, "MeshCNN", "meshcnn_dfm_best.pt")
}

DEVICE = torch.device("cpu")  # CPU-only for web app inference

# Module-level model cache
_model_cache = {}

def _init_models():
    """Instantiate and load weights for all models."""
    models = {}
    
    # DiffusionNet (For Volume)
    try:
        model_dn = DFM_DiffusionNet(in_dim=7, C=128, n_blocks=4, graph_stat_dim=11, dropout=0.3, n_reg=1, n_clf=5).to(DEVICE)
        model_dn.load_state_dict(torch.load(_WEIGHTS_PATHS['DiffusionNet'], map_location=DEVICE, weights_only=False))
        model_dn.eval()
        models['DiffusionNet'] = model_dn
    except Exception as e:
        print(f"Failed to load DiffusionNet: {e}")

    # NodeLevel DiffusionNet (For Saliency & Constraints)
    try:
        model_node = NodeLevel_DFM_DiffusionNet(in_dim=7, C=128, n_blocks=4, graph_stat_dim=11, dropout=0.3, n_clf=5).to(DEVICE)
        model_node.load_state_dict(torch.load(_WEIGHTS_PATHS['NodeLevel_DiffusionNet'], map_location=DEVICE, weights_only=False))
        model_node.eval()
        models['NodeLevel_DiffusionNet'] = model_node
    except Exception as e:
        print(f"Failed to load NodeLevel_DiffusionNet: {e}")

    # GAT
    try:
        model_gat = DFM_GAT(in_dim=9, hidden_dim=64, graph_stat_dim=11, dropout=0.2, n_reg=1, n_clf=5).to(DEVICE)
        model_gat.load_state_dict(torch.load(_WEIGHTS_PATHS['GAT'], map_location=DEVICE, weights_only=False))
        model_gat.eval()
        models['GAT'] = model_gat
    except Exception as e:
        print(f"Failed to load GAT: {e}")

    # MeshCNN
    try:
        model_meshcnn = DFM_MeshCNN(in_ch=5, hidden_ch=64, graph_stat_dim=11, dropout=0.3, pool_sizes=(512, 256), n_reg=1, n_clf=5).to(DEVICE)
        # Note: adjust loading if 'model_state_dict' structure was used
        state_dict = torch.load(_WEIGHTS_PATHS['MeshCNN'], map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in state_dict:
             state_dict = state_dict['model_state_dict']
        model_meshcnn.load_state_dict(state_dict)
        model_meshcnn.eval()
        models['MeshCNN'] = model_meshcnn
    except Exception as e:
        print(f"Failed to load MeshCNN: {e}")

    # PointNet++
    try:
        model_pn = DFM_PointNetPP(in_dim=6, graph_stat_dim=11, dropout=0.3).to(DEVICE)
        model_pn.load_state_dict(torch.load(_WEIGHTS_PATHS['PointNet++'], map_location=DEVICE, weights_only=False))
        model_pn.eval()
        models['PointNet++'] = model_pn
    except Exception as e:
        print(f"Failed to load PointNet++: {e}")

    return models


def get_models():
    """Return cached models, loading on first call."""
    global _model_cache
    if not _model_cache:
        _model_cache = _init_models()
    return _model_cache


def run_inference(stl_path_or_file) -> dict:
    """
    Run the dual-model inference (DiffusionNet for volume, NodeLevel for saliency/constraints).
    """
    data, raw_mesh, mesh_info = process_stl_for_inference(stl_path_or_file)
    models = get_models()
    model_vol = models.get('DiffusionNet')
    model_node = models.get('NodeLevel_DiffusionNet')
    
    if model_vol is None or model_node is None:
        raise ValueError("One or both DiffusionNet models failed to load.")
        
    data = data.to(DEVICE)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)

    # 1. Volume & Constraints (Original Graph-Level Model)
    with torch.no_grad():
        pred_reg, pred_clf = model_vol(data)

    log_vol = pred_reg.item()
    volume = float(np.expm1(log_vol))

    probs = torch.sigmoid(pred_clf).squeeze().cpu().numpy()
    constraints = {}
    for i, name in enumerate(CLF_NAMES):
        p = float(probs[i])
        constraints[name] = {
            'passed': p > 0.5,
            'confidence': round(p * 100, 1),
        }

    # 2. Saliency Maps (Node-Level Model)
    with torch.no_grad():
        node_logits = model_node(data)
        node_probs = torch.sigmoid(node_logits).cpu().numpy()  # [N, 5]

    saliency_maps = {}
    for i, name in enumerate(CLF_NAMES):
        saliency_maps[name] = node_probs[:, i].tolist()

    return {
        'predicted_log_volume': round(log_vol, 4),
        'predicted_volume_mm3': round(volume, 2),
        'constraints': constraints,
        'mesh_info': mesh_info,
        'raw_mesh': raw_mesh,
        'saliency_maps': saliency_maps,
        'data_obj': data # Pass this so we don't re-process for comparative runs
    }


def _move_data_to_device(data, device):
    """Move all tensor attributes of SimpleData to device."""
    data = data.to(device)
    # Explicitly move extras that .to() may miss due to SimpleData implementation
    for attr in ['gat_x', 'pos', 'edge_feats', 'e2e',
                 'evals', 'evecs', 'mass',
                 'grad_rows', 'grad_cols', 'grad_vals', 'grad_n']:
        if hasattr(data, attr) and isinstance(getattr(data, attr), torch.Tensor):
            setattr(data, attr, getattr(data, attr).to(device))
    return data


def run_comparative_inference(data) -> dict:
    """
    Run all loaded models on the preprocessed Data object to compare metrics.
    Returns dict keyed by model name, with Volume + per-constraint PASS/FAIL.
    """
    models = get_models()
    results = {}

    data = _move_data_to_device(data, DEVICE)
    N = data.x.shape[0]
    data.batch = torch.zeros(N, dtype=torch.long, device=DEVICE)

    # Column names must match what render_comp_dashboard expects in app.py
    clf_display = {
        'area':           'Area',
        'contour_count':  'Contour Count',
        'contour_length': 'Contour Length',
        'overhang':       'Overhang',
        'pass_fail':      'Pass Fail',
    }

    for name, model in models.items():
        try:
            with torch.no_grad():
                pred_reg, pred_clf = model(data)

            log_vol = pred_reg.item()
            volume  = float(np.expm1(log_vol))
            probs   = torch.sigmoid(pred_clf).squeeze().cpu().numpy()

            row = {'Volume (mm\u00b3)': f"{volume:,.1f}"}
            for i, internal_name in enumerate(CLF_NAMES):
                display_name = clf_display.get(internal_name, internal_name)
                row[display_name] = "PASS" if float(probs[i]) > 0.5 else "FAIL"

            results[name] = row

        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = {k: "ERR" for k in
                             ['Volume (mm\u00b3)', 'Area', 'Contour Count',
                              'Contour Length', 'Overhang', 'Pass Fail']}
            results[name]['_error'] = str(e)

    return results
