# final_streamlit_yolo_seg_voc.py

# ==========================================
# HEADLESS MODE - MUST BE FIRST
# ==========================================
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import numpy as np
import streamlit as st
import torch
import torch.serialization
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import time
import cv2
import tempfile
import gdown

# ==========================================
# 1. CONFIGURATION & VOC METADATA
# ==========================================

MODEL_REGISTRY = {
    "UNet": {
        "url": "https://drive.google.com/uc?id=1s18h16cFLz29owtEfb2lj0n3GsCBb0uy",
        "model_type": "unet",
        "img_size": 384,
    },
    "DeepLabV3": {
        "url": "https://drive.google.com/uc?id=1XeMRUTA47E4QICTAIVe-eB8nI0i5CIl7",
        "model_type": "deeplabv3",
        "img_size": 384,
    },
    "YOLO": {
        "url": "https://drive.google.com/uc?id=1GXfUEkDScw9zveLd9xOYrhOCSsz6cdCd",
        "model_type": "yolo",
        "img_size": 640,
    },
}

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ==========================================
# 2. XAI UTILITIES (ACTIVATION MAPPING)
# ==========================================
class ActivationHook:
    """Captures activations from a specific layer."""
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.detach()

    def remove(self):
        self.hook.remove()

def generate_heatmap(activation_tensor):
    """Converts 4D tensor to a normalized 2D heatmap."""
    heatmap = torch.mean(activation_tensor, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
voc_colors = np.array([
    [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128], [0,128,128], [128,128,128],
    [64,0,0], [192,0,0], [64,128,0], [192,128,0], [64,0,128], [192,0,128], [64,128,128], [192,128,128],
    [0,64,0], [128,64,0], [0,192,0], [128,192,0], [0,64,128]
], dtype=np.uint8)

def pascal_voc_palette():
    return voc_colors

def plot_yolo_with_voc_colors(results, img):
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        # guard index
        cls_id = max(0, min(cls_id, len(voc_colors)-1))
        color = tuple(map(int, voc_colors[cls_id]))
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(img,
                    results[0].names.get(cls_id, str(cls_id)),
                    (xyxy[0], max(12, xyxy[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1)
    return img

def decode_voc_mask(mask):
    return voc_colors[np.clip(mask, 0, 20)]

def show_voc_legend():
    st.subheader("VOC Color Legend")
    legend_cols = st.columns(5)
    for i, cls in enumerate(VOC_CLASSES):
        color = voc_colors[i].tolist()
        patch = np.zeros((30, 30, 3), dtype=np.uint8)
        patch[:] = color
        with legend_cols[i % 5]:
            st.image(patch, width='content')
            st.caption(cls)

def draw_voc_legend_on_image(img, detected_cls_ids):
    legend_img = img.copy()
    h, w, _ = legend_img.shape
    start_x, start_y = w - 330, 10
    box_size = 15
    spacing = 20
    for i, cls_id in enumerate(sorted(set(detected_cls_ids))):
        if cls_id < 0 or cls_id >= len(VOC_CLASSES):
            continue
        color = tuple(map(int, voc_colors[cls_id]))
        y = start_y + i * spacing
        cv2.rectangle(legend_img, (start_x, y), (start_x + box_size, y + box_size), color, -1)
        cv2.putText(legend_img, VOC_CLASSES[cls_id], (start_x + box_size + 10, y + box_size -3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return legend_img

def preprocess_for_yolo(img_pil, size=384):
    img_pil = img_pil.convert("RGB")
    w, h = img_pil.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    img_padded = Image.new("RGB", (size, size), (0, 0, 0))
    img_padded.paste(img_resized, ((size - new_w) // 2, (size - new_h) // 2))
    return img_padded

def preprocess_pil(img_pil, size=384):
    img_pil = img_pil.convert("RGB")
    w, h = img_pil.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
    img_padded = Image.new("RGB", (size, size), (0, 0, 0))
    img_padded.paste(img_resized, ((size - new_w) // 2, (size - new_h) // 2))
    x = TF.to_tensor(img_padded)
    x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return x.unsqueeze(0), img_padded

def make_overlay(img, mask, alpha=0.45):
    img, mask = img.astype(np.float32), mask.astype(np.float32)
    return np.clip((1 - alpha) * img + alpha * mask, 0, 255).astype(np.uint8)

# ==========================================
# 4. HYBRID LOADER
# ==========================================
def download_checkpoint(source, is_yolo=False):
    if not source.startswith("http"):
        return source
    suffix = ".pt" if is_yolo else ".pth"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        temp_path = f.name
    gdown.download(source, temp_path, quiet=True)
    return temp_path

@st.cache_resource
def load_model(ckpt_path, device="cpu", model_name=None):
    try:
        from torchvision.models.segmentation.deeplabv3 import DeepLabV3
        torch.serialization.add_safe_globals([DeepLabV3])
    except: pass

    model_type = (model_name or "").lower()
    is_yolo = model_type == "yolo" or "yolo" in ckpt_path.lower()
    ckpt_path = download_checkpoint(ckpt_path, is_yolo=is_yolo)

    if is_yolo:
        model = YOLO(ckpt_path)
        try:
            model.task = "segment"
        except Exception:
            pass
        model.model.names = {i: name for i, name in enumerate(VOC_CLASSES)}
        return model, {"model_type": "yolo", "img_size": 640}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval(), {"model_type": "deeplabv3", "img_size": 384}

    if isinstance(ckpt, dict):
        cfg = ckpt.get("cfg", {})
        state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        cfg = {}
        state = ckpt

    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    encoder = ckpt.get("encoder") if isinstance(ckpt, dict) else None
    encoder = encoder or cfg.get("encoder", "resnet50")
    num_classes = ckpt.get("num_classes") if isinstance(ckpt, dict) else None
    num_classes = num_classes or cfg.get("num_classes", 21)
    img_size = ckpt.get("img_size") if isinstance(ckpt, dict) else None
    img_size = img_size or cfg.get("img_size", 384)
    m_type = str(ckpt.get("model_type") or cfg.get("model_type") or "unet").lower() if isinstance(ckpt, dict) else "unet"

    model_cls = smp.DeepLabV3Plus if "deeplab" in m_type else smp.Unet
    model = model_cls(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=num_classes).to(device)
    model.load_state_dict(state)
    return model.eval(), {"model_type": m_type, "img_size": img_size}

#==================== added for yolo segmentation visualization with VOC colors ====================
def yolo_instances_to_class_mask(results, target_h, target_w):
    """Convert YOLO instance masks to a semantic class mask (H,W)."""
    res = results[0]
    masks = None
    classes = []
    if hasattr(res, "masks") and res.masks is not None:
        masks_data = res.masks.data
        masks = masks_data.cpu().numpy() if hasattr(masks_data, "cpu") else np.array(masks_data)
    if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
        try:
            classes = res.boxes.cls.cpu().numpy().astype(int).tolist()
        except Exception:
            classes = [int(x) for x in res.boxes.cls]

    if masks is None or len(masks) == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.shape[1] != target_h or masks.shape[2] != target_w:
        resized = []
        for m in masks:
            m_resized = cv2.resize((m.astype(np.uint8) * 255), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            resized.append((m_resized > 127).astype(np.uint8))
        masks = np.stack(resized, axis=0)

    class_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    for i, m in enumerate(masks):
        cls_id = classes[i] if i < len(classes) else 0
        if cls_id < 0 or cls_id >= len(VOC_CLASSES):
            continue
        class_mask[m.astype(bool)] = cls_id
    return class_mask

def plot_yolo_semantic(results, img_padded, alpha=0.6):
    """Produce UNet/DeepLab‑style overlay for YOLO segmentation."""
    img_np = np.array(img_padded) if hasattr(img_padded, "convert") else img_padded.copy()
    H, W = img_np.shape[:2]
    class_mask = yolo_instances_to_class_mask(results, H, W)
    colored = decode_voc_mask(class_mask)
    blended = make_overlay(img_np, colored, alpha=alpha)
    present_classes = np.unique(class_mask)
    blended = draw_voc_legend_on_image(blended, present_classes)
    return blended

def masks_to_colored_overlay(masks, classes, img_shape, palette=voc_colors, alpha=0.6):
    """
    masks: boolean array (N, H, W) or list of masks
    classes: list/array of class ids per mask (len N)
    img_shape: (H, W, 3) of the display image
    Returns: overlay image (H,W,3) with colored masks blended
    """
    H, W = img_shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    combined_alpha = np.zeros((H, W), dtype=np.float32)

    for i, m in enumerate(masks):
        cls_id = int(classes[i]) if i < len(classes) else 0
        if cls_id < 0 or cls_id >= len(palette):
            cls_id = 0
        color = tuple(map(int, palette[cls_id]))
        mask_uint8 = (m.astype(np.uint8) * 255).astype(np.uint8)
        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[:, :, 0] = (mask_uint8 * color[0]) // 255
        color_layer[:, :, 1] = (mask_uint8 * color[1]) // 255
        color_layer[:, :, 2] = (mask_uint8 * color[2]) // 255

        alpha_mask = (mask_uint8 / 255.0) * (1.0 - combined_alpha)
        for c in range(3):
            overlay[:, :, c] = (overlay[:, :, c].astype(np.float32) + color_layer[:, :, c].astype(np.float32) * alpha_mask).astype(np.uint8)
        combined_alpha = np.clip(combined_alpha + alpha_mask, 0, 1.0)

    return overlay, combined_alpha

def plot_yolo_segmentation(results, img_padded, palette=voc_colors, alpha=0.6):
    """
    results: ultralytics results from model.predict(..., task='segment')
    img_padded: PIL.Image or numpy array (RGB) of the padded input used for prediction
    """
    if hasattr(img_padded, "convert"):
        img_np = np.array(img_padded)
    else:
        img_np = img_padded.copy()

    H, W = img_np.shape[:2]
    res = results[0]

    masks = None
    classes = []
    try:
        if hasattr(res, "masks") and res.masks is not None:
            masks_data = res.masks.data
            if isinstance(masks_data, np.ndarray):
                masks = masks_data
            else:
                masks = masks_data.cpu().numpy()
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            try:
                classes = [int(x) for x in res.boxes.cls.cpu().numpy().astype(int)]
            except Exception:
                classes = [int(x) for x in res.boxes.cls]
    except Exception:
        masks = None

    if masks is not None and len(masks) > 0:
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        if masks.shape[1] != H or masks.shape[2] != W:
            resized_masks = []
            for m in masks:
                m_resized = cv2.resize((m.astype(np.uint8) * 255), (W, H), interpolation=cv2.INTER_NEAREST)
                resized_masks.append((m_resized > 127).astype(np.uint8))
            masks = np.stack(resized_masks, axis=0)

        overlay_colored, combined_alpha = masks_to_colored_overlay(masks, classes, img_np.shape, palette=palette, alpha=alpha)
        alpha_total = np.expand_dims(combined_alpha * alpha, axis=-1)
        blended = (img_np.astype(np.float32) * (1 - alpha_total) + overlay_colored.astype(np.float32) * alpha_total).astype(np.uint8)

        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                cls_id = classes[i] if i < len(classes) else 0
                if cls_id < 0 or cls_id >= len(VOC_CLASSES):
                    cls_id = 0
                color = tuple(map(int, palette[cls_id]))
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.putText(blended, VOC_CLASSES[cls_id], (max(0, cx-20), max(12, cy-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        return blended
    else:
        return img_np

# ==========================================
# 5. UI & INFERENCE
# ==========================================
st.set_page_config(page_title="Architectural Analysis", layout="wide")
st.title("🚀 VOC Model Benchmarking & Architectural Analysis")

st.sidebar.header("Configuration")
selected_models = [m for m in MODEL_REGISTRY.keys() if st.sidebar.checkbox(m, value=True)]
device_choice = st.sidebar.selectbox("Compute Device", ["cpu", "cuda"], index=0)
alpha = st.sidebar.slider("Mask Opacity", 0.0, 1.0, 0.65)
show_xai = st.sidebar.toggle("Enable Architectural Visualization", value=False)

uploaded = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"])

if uploaded and selected_models:
    img_pil = Image.open(uploaded).convert("RGB")
    st.image(np.array(img_pil), caption="Original", width='stretch')
    st.divider()

    cols = st.columns(len(selected_models))

    for idx, name in enumerate(selected_models):
        model_info = MODEL_REGISTRY[name]
        path = model_info["url"]

        with st.status(f"Running {name}...", expanded=False):
            model, meta = load_model(path, device=device_choice, model_name=name)

            # Setup Hook for XAI (Targeting ResNet50 deepest encoder stage)
            hook = None
            if show_xai and name == "UNet":
                hook = ActivationHook(model.encoder.layer4)

            if meta["model_type"] == "yolo":
                t_start = time.perf_counter()
                img_padded = preprocess_for_yolo(img_pil, size=meta["img_size"])
                results = model.predict(img_padded, imgsz=meta["img_size"], device=device_choice, task="segment", verbose=False)
                inference_time = (time.perf_counter() - t_start) * 1000

                try:
                    #img_display = plot_yolo_segmentation(results, img_padded, palette=voc_colors, alpha=alpha)
                    img_display = plot_yolo_semantic(results, img_padded, alpha=alpha)
                    
                except Exception:
                    img_display = np.array(img_padded).copy()
                    img_display = plot_yolo_with_voc_colors(results, img_display)

            else:
                x, img_padded = preprocess_pil(img_pil, size=meta["img_size"])
                x = x.to(device_choice)
                t_start = time.perf_counter()
                with torch.inference_mode():
                    out = model(x)
                    logits = out["out"] if isinstance(out, dict) else out
                    pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
                inference_time = (time.perf_counter() - t_start) * 1000
                img_display = make_overlay(np.array(img_padded), decode_voc_mask(pred.squeeze(0)), alpha=alpha)
                pred_classes = np.unique(pred)
                img_display = draw_voc_legend_on_image(img_display, pred_classes)

            # Process XAI if enabled
            heatmap = None
            if hook and hook.features is not None:
                heatmap = generate_heatmap(hook.features)
                hook.remove()

            fps = 1000 / inference_time if inference_time > 0 else 0

        with cols[idx]:
            st.subheader(name)
            st.image(img_display, width='stretch')

            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Latency", f"{inference_time:.1f} ms")
            m_col2.metric("FPS", f"{fps:.1f}")

            if show_xai and name == "UNet" and heatmap is not None:
                with st.expander("🔍 Feature Activations"):
                    st.image(heatmap, caption="ResNet-50 Encoder Stage 4", width='stretch')
                    st.caption("Warm colors (red) represent areas of strong semantic focus.")
