import os
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

# ==========================================
# 1. CONFIGURATION & VOC METADATA
# ==========================================
MODEL_REGISTRY = {
    "UNet": "ckpt_voc_demo_unet.pth",
    "DeepLabV3": "ckpt_voc_demo_deeplabv3.pth",
    "YOLO": "ckpt_voc_demo_yolo_seg.pt", 
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
    # Mean over channels to get spatial importance
    heatmap = torch.mean(activation_tensor, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    # Apply colormap for better visualization
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

def plot_yolo_with_voc_colors(results, img):
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        color = tuple(map(int, voc_colors[cls_id+1]))  # +1 to skip background
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cv2.rectangle(img, xyxy[:2], xyxy[2:], color, 2)
        cv2.putText(img, 
                    results[0].names[cls_id], 
                    xyxy[:2], 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,     # font scale
                    color, 
                    2)       # thicker text for better visibility
    return img

def decode_voc_mask(mask):
    return voc_colors[np.clip(mask, 0, 20)]

def show_voc_legend():
    st.subheader("VOC Color Legend")
    legend_cols = st.columns(5)  # adjust number of columns
    for i, cls in enumerate(VOC_CLASSES):
        color = voc_colors[i].tolist()
        # Create a small color patch
        patch = np.zeros((30, 30, 3), dtype=np.uint8)
        patch[:] = color
        with legend_cols[i % 5]:
            st.image(patch, use_container_width=False)
            st.caption(cls)

def draw_voc_legend_on_image(img, detected_cls_ids):
    # Create a copy so we don’t overwrite
    legend_img = img.copy()
    h, w, _ = legend_img.shape

    # Legend placement (bottom-right corner)
    start_x, start_y = w - 330, 10 # Starting point for legend
    box_size = 15
    spacing = 20

    for i, cls_id in enumerate(sorted(set(detected_cls_ids))):
        color = tuple(map(int, voc_colors[cls_id]))
        y = start_y + i * spacing

        # Draw color box
        cv2.rectangle(legend_img,
                      (start_x, y),
                      (start_x + box_size, y + box_size),
                      color, -1)

        # Draw class name
        cv2.putText(legend_img,
                    VOC_CLASSES[cls_id],
                    (start_x + box_size + 10, y + box_size -3 ), # Adjust text position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,255), 1) # White text for contrast

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
@st.cache_resource
def load_model(ckpt_path, device="cpu"):
    try:
        from torchvision.models.segmentation.deeplabv3 import DeepLabV3
        torch.serialization.add_safe_globals([DeepLabV3])
    except: pass

    if "yolo" in ckpt_path.lower():
        # Load YOLO segmentation checkpoint directly using Ultralytics API
        # Use YOLO(...) with the checkpoint path; specify task='segment' for clarity
        model = YOLO(ckpt_path)            # loads weights and config
        # Ensure model is configured for segmentation
        try:
            # If the model object has a 'task' attribute, set it to 'segment' for clarity
            model.task = "segment"
        except Exception:
            pass

        # Ensure names include background and all VOC classes
        # Ultralytics expects names mapping for class indices (0..20)
        model.model.names = {i: name for i, name in enumerate(VOC_CLASSES)}
        return model, {"model_type": "yolo", "img_size": 640}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval(), {"model_type": "deeplabv3", "img_size": 384}
        
    cfg = ckpt.get("cfg", {})
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    encoder = ckpt.get("encoder") or cfg.get("encoder", "resnet50")
    num_classes = ckpt.get("num_classes") or cfg.get("num_classes", 21)
    img_size = ckpt.get("img_size") or cfg.get("img_size", 384)
    m_type = str(ckpt.get("model_type") or cfg.get("model_type") or "unet").lower()

    model_cls = smp.DeepLabV3Plus if "deeplab" in m_type else smp.Unet
    model = model_cls(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=num_classes).to(device)
    if hasattr(state, 'state_dict'): state = state.state_dict()
    model.load_state_dict(state)
    return model.eval(), {"model_type": m_type, "img_size": img_size}

#==================== added for yolo segmentation visualization with VOC colors ====================
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

    # iterate instances; later instances will overlay earlier ones
    for i, m in enumerate(masks):
        cls_id = int(classes[i])
        color = tuple(map(int, palette[cls_id]))  # palette indexed by class id
        mask_uint8 = (m.astype(np.uint8) * 255).astype(np.uint8)
        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[:, :, 0] = mask_uint8 * color[0] // 255
        color_layer[:, :, 1] = mask_uint8 * color[1] // 255
        color_layer[:, :, 2] = mask_uint8 * color[2] // 255

        # accumulate weighted color and alpha
        alpha_mask = (mask_uint8 / 255.0) * (1.0 - combined_alpha)
        for c in range(3):
            overlay[:, :, c] = (overlay[:, :, c].astype(np.float32) + color_layer[:, :, c].astype(np.float32) * alpha_mask).astype(np.uint8)
        combined_alpha = np.clip(combined_alpha + alpha_mask, 0, 1.0)

    # blend overlay with original image outside this function
    return overlay, combined_alpha

def plot_yolo_segmentation(results, img_padded, palette=voc_colors, alpha=0.6):
    """
    results: ultralytics results from model.predict(..., task='segment')
    img_padded: PIL.Image or numpy array (RGB) of the padded input used for prediction
    """
    # convert to numpy RGB
    if hasattr(img_padded, "convert"):
        img_np = np.array(img_padded)
    else:
        img_np = img_padded.copy()

    H, W = img_np.shape[:2]
    res = results[0]

    # Extract masks and classes robustly
    masks = None
    classes = []
    try:
        # Ultralytics returns masks in res.masks.data (boolean) and res.boxes.cls for classes
        if hasattr(res, "masks") and res.masks is not None:
            # res.masks.data shape: (N, H, W) or (N, H*W) depending on version
            masks_data = res.masks.data
            # ensure boolean masks in shape (N,H,W)
            if isinstance(masks_data, np.ndarray):
                masks = masks_data
            else:
                # torch tensor
                masks = masks_data.cpu().numpy()
        # classes: res.boxes.cls or res.boxes.cls.cpu().numpy()
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            try:
                classes = [int(x) for x in res.boxes.cls.cpu().numpy().astype(int)]
            except Exception:
                # fallback: use res.boxes.cls as list
                classes = [int(x) for x in res.boxes.cls]
    except Exception:
        masks = None

    # If masks exist, rasterize and overlay
    if masks is not None and len(masks) > 0:
        # ensure masks shape (N,H,W)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        # If masks are smaller/larger than img, try to resize each mask to img size
        if masks.shape[1] != H or masks.shape[2] != W:
            resized_masks = []
            for m in masks:
                m_resized = cv2.resize((m.astype(np.uint8) * 255), (W, H), interpolation=cv2.INTER_NEAREST)
                resized_masks.append((m_resized > 127).astype(np.uint8))
            masks = np.stack(resized_masks, axis=0)

        overlay_colored, combined_alpha = masks_to_colored_overlay(masks, classes, img_np.shape, palette=palette, alpha=alpha)
        # blend: blended = (1-alpha_total)*img + alpha_total*overlay_colored
        alpha_total = np.expand_dims(combined_alpha * alpha, axis=-1)
        blended = (img_np.astype(np.float32) * (1 - alpha_total) + overlay_colored.astype(np.float32) * alpha_total).astype(np.uint8)

        # draw instance labels (class names) at box centers if boxes exist
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                cls_id = classes[i] if i < len(classes) else 0
                color = tuple(map(int, palette[cls_id]))
                x1, y1, x2, y2 = box.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.putText(blended, VOC_CLASSES[cls_id], (max(0, cx-20), max(12, cy-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        return blended
    else:
        # fallback: return original image if no masks
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
    st.image(np.array(img_pil), caption="Original", use_container_width=True)
    st.divider()

    cols = st.columns(len(selected_models))

    for idx, name in enumerate(selected_models):
        path = MODEL_REGISTRY[name]
        if not os.path.exists(path):
            cols[idx].error(f"Missing: {name}")
            continue

        with st.status(f"Running {name}...", expanded=False):
            model, meta = load_model(path, device=device_choice)
            
            # Setup Hook for XAI (Targeting ResNet50 deepest encoder stage)
            hook = None
            if show_xai and name == "UNet":
                hook = ActivationHook(model.encoder.layer4)

            # if meta["model_type"] == "yolo":
            #     t_start = time.perf_counter()
            #     img_padded = preprocess_for_yolo(img_pil, size=meta["img_size"])
            #     results = model.predict(img_padded, imgsz=meta["img_size"], device=device_choice, verbose=False)
            #     inference_time = (time.perf_counter() - t_start) * 1000
            #     img_display = np.array(img_padded).copy()
            #     img_display = plot_yolo_with_voc_colors(results, img_display)
            if meta["model_type"] == "yolo":
                t_start = time.perf_counter()
                img_padded = preprocess_for_yolo(img_pil, size=meta["img_size"])
                # Ultralytics accepts PIL.Image or path; pass task='segment' explicitly
                results = model.predict(img_padded, imgsz=meta["img_size"], device=device_choice, task="segment", verbose=False)
                inference_time = (time.perf_counter() - t_start) * 1000

                # Visualize segmentation masks (preferred) or fallback to boxes
                try:
                    img_display = plot_yolo_segmentation(results, img_padded, palette=voc_colors, alpha=alpha)
                except Exception as e:
                    # fallback to original plotting (boxes) if segmentation plotting fails
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

            # Calculate FPS
            fps = 1000 / inference_time if inference_time > 0 else 0

        with cols[idx]:
            st.subheader(name)
            st.image(img_display, use_container_width=True)
            #show_voc_legend()
            
            # --- METRICS ROW ---
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Latency", f"{inference_time:.1f} ms")
            m_col2.metric("FPS", f"{fps:.1f}")
            
            if show_xai and name == "UNet" and heatmap is not None:
                with st.expander("🔍 Feature Activations"):
                    st.image(heatmap, caption="ResNet-50 Encoder Stage 4", use_container_width=True)
                    st.caption("Warm colors (red) represent areas of strong semantic focus.")