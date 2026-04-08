"""
Inference pipeline: load DiffusionNet model and run predictions on STL files.
"""

import os
import torch
import numpy as np

from models.diffusionnet import DFM_DiffusionNet
from preprocessing.mesh_to_graph import process_stl_for_inference, CLF_NAMES

# Default path to trained weights (relative to project root)
_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "DiffusionNet",
    "diffusionnet_dfm_best_5thApril.pt",
)

DEVICE = torch.device("cpu")  # CPU-only for web app inference


def _load_model(weights_path: str = _DEFAULT_WEIGHTS) -> DFM_DiffusionNet:
    """Instantiate the model and load trained weights."""
    model = DFM_DiffusionNet(
        in_dim=7,
        C=128,
        n_blocks=4,
        graph_stat_dim=11,
        dropout=0.3,
        n_reg=1,
        n_clf=5,
    ).to(DEVICE)

    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


# Module-level model cache (loaded once)
_model_cache = None


def get_model() -> DFM_DiffusionNet:
    """Return cached model, loading on first call."""
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model()
    return _model_cache


def compute_saliency_map(model, data, clf_idx):
    """Compute GradCAM saliency for a specific classification head."""
    model.eval()
    
    grad_holder = {}
    def save_grad(grad):
        grad_holder['g'] = grad

    pred_reg, pred_clf, node_acts = model.forward_gradcam(data)
    node_acts.register_hook(save_grad)

    scalar = pred_clf[0, clf_idx]
    
    model.zero_grad()
    scalar.backward()

    A = node_acts.detach().cpu().numpy()
    dSdA = grad_holder['g'].detach().cpu().numpy()

    alpha = dSdA.mean(axis=0)
    cam = (A * alpha[np.newaxis, :]).sum(axis=1)
    cam = np.maximum(cam, 0)
    
    cam = np.log1p(cam)
    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)
        
    return cam

def run_inference(stl_path_or_file) -> dict:
    """
    End-to-end inference on an STL file.

    Parameters
    ----------
    stl_path_or_file : str or file-like
        Path to STL or an uploaded file.

    Returns
    -------
    dict with keys:
        predicted_log_volume  : float
        predicted_volume_mm3  : float  (expm1 of log volume)
        constraints           : dict[str, dict] with 'passed' (bool) and 'confidence' (float)
        mesh_info             : dict   raw mesh properties
        saliency_maps         : dict   of metric names to numpy arrays
    """
    # 1. Preprocess
    data, raw_mesh, mesh_info = process_stl_for_inference(stl_path_or_file)

    # 2. Run model
    model = get_model()
    data = data.to(DEVICE)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pred_reg, pred_clf = model(data)

    # 3. Post-process regression → volume
    log_vol = pred_reg.item()
    volume = float(np.expm1(log_vol))  # undo log1p

    # 4. Post-process classification → pass/fail + confidence
    probs = torch.sigmoid(pred_clf).squeeze().cpu().numpy()
    constraints = {}
    for i, name in enumerate(CLF_NAMES):
        p = float(probs[i])
        constraints[name] = {
            'passed': p > 0.5,
            'confidence': round(p * 100, 1),
        }

    # 5. Compute Saliency Maps for constraints
    saliency_maps = {}
    for i, name in enumerate(CLF_NAMES):
        # Setting requires_grad on necessary tensors just for this
        data.x.requires_grad_()
        saliency_maps[name] = compute_saliency_map(model, data, i).tolist() # Convert to list for easier passing

    return {
        'predicted_log_volume': round(log_vol, 4),
        'predicted_volume_mm3': round(volume, 2),
        'constraints': constraints,
        'mesh_info': mesh_info,
        'raw_mesh': raw_mesh,
        'saliency_maps': saliency_maps,
    }
