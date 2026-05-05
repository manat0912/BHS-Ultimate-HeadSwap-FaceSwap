import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import v2
from skimage import transform as trans

# ── Qt bootstrap ──────────────────────────────────────────────────────────────
from PySide6 import QtCore
_qt_app = None
def _ensure_qt_app():
    global _qt_app
    if QtCore.QCoreApplication.instance() is None:
        _qt_app = QtCore.QCoreApplication(sys.argv)
_ensure_qt_app()

# ── Add VisoMaster to sys.path ────────────────────────────────────────────────
VISO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "VisoMaster"))
if VISO_DIR not in sys.path:
    sys.path.insert(0, VISO_DIR)

import onnxruntime as ort
from app.processors.models_processor import ModelsProcessor
from app.processors.utils import faceutil
from app.processors.models_data import models_dir as _VM_MODELS_DIR_REL

# ── MockMainWindow ────────────────────────────────────────────────────────────
class _MockMainWindow:
    class _Signal:
        def emit(self, *a, **kw): pass
    class _Button:
        def __init__(self): self._checked = False
        def isChecked(self): return self._checked
        def setChecked(self, v): self._checked = v
    def __init__(self):
        self.model_loading_signal = self._Signal()
        self.model_loaded_signal  = self._Signal()
        self.model_load_dialog    = None
        self.parameters           = {}
        self.control              = {'DetectorModelSelection': 'RetinaFace', 'MaxDFMModelsSlider': 1}
        self.default_parameters   = {'SwapModelSelection': 'Inswapper128'}
        self.target_faces         = {}
        self.dfm_models_data      = {}
        self.editFacesButton      = self._Button()
        self.swapfacesButton      = self._Button()
        self.faceCompareCheckBox  = self._Button()
        self.faceMaskCheckBox     = self._Button()

_SWAPPER_MODEL_MAP = {
    'Inswapper128': 'Inswapper128', 'SimSwap512': 'SimSwap512',
    'GhostFace-v1': 'GhostFace-v1', 'GhostFace-v2': 'GhostFace-v2',
    'GhostFace-v3': 'GhostFace-v3', 'CSCS': 'CSCS',
}
_ARCFACE_MAP = {
    'Inswapper128': 'Inswapper128ArcFace', 'SimSwap512': 'SimSwapArcFace',
    'GhostFace-v1': 'GhostArcFace', 'GhostFace-v2': 'GhostArcFace',
    'GhostFace-v3': 'GhostArcFace', 'CSCS': 'CSCSArcFace',
}
RESTORER_CHOICES = ['None', 'GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'RestoreFormer++']


def _make_oval_mask(size: int, device, inner: float = 0.60) -> torch.Tensor:
    """
    Return a (1, size, size) float32 tensor with a smooth oval falloff:
      1.0 inside the inner ellipse, fading smoothly to 0 at the edge.
    This prevents any rectangular box artifact when pasting back.
    """
    c = size / 2.0
    # rx slightly smaller than ry for a natural face-shaped oval
    rx = c * 0.92
    ry = c * 0.98
    ys = torch.arange(size, dtype=torch.float32, device=device)
    xs = torch.arange(size, dtype=torch.float32, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    dist = ((X - c) / rx) ** 2 + ((Y - c) / ry) ** 2   # 0=center, 1=rim
    # Smooth transition from inner→1.0 boundary
    mask = ((1.0 - dist) / (1.0 - inner)).clamp(0.0, 1.0)
    mask = torch.where(dist < inner, torch.ones_like(mask), mask)
    return mask.unsqueeze(0)   # (1, H, W)


def _paste_back(enhanced: torch.Tensor,
                target: torch.Tensor,
                tform_inv,
                device: str,
                oval_mask: torch.Tensor) -> torch.Tensor:
    """
    Inverse-affine the enhanced 512×512 face crop back into the full-frame
    target tensor using a pre-built oval mask for seamless blending.
    """
    H, W = target.shape[1], target.shape[2]

    # Pad enhanced & mask to full-frame size, then inverse-warp
    def _iaffine(t):
        p = v2.functional.pad(t, (0, 0, W - 512, H - 512))
        return v2.functional.affine(
            p,
            tform_inv.rotation * 57.2958,
            (tform_inv.translation[0], tform_inv.translation[1]),
            tform_inv.scale, 0,
            interpolation=v2.InterpolationMode.BILINEAR,
            center=(0, 0),
        )

    enh_full  = _iaffine(enhanced)
    mask_full = _iaffine(oval_mask)

    return (enh_full * mask_full + target * (1.0 - mask_full)).clamp(0, 255)


class VisoBridge:
    def __init__(self, device: str = 'cuda'):
        _ensure_qt_app()
        if device == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("[VISO] CUDA unavailable – using CPU.")
            device = 'cpu'
        self._device = device
        self._mock_win = _MockMainWindow()
        print(f"[VISO] Initialising ModelsProcessor on {device} …")
        self.processor = ModelsProcessor(self._mock_win, device=device)
        self._patch_model_paths()
        self._ensure_detector()
        # Pre-build the oval paste mask (reused every frame)
        self._oval_512 = _make_oval_mask(512, device)
        print("[VISO] Ready.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _patch_model_paths(self):
        for d in [self.processor.models_path]:
            for k, v in list(d.items()):
                if v and not os.path.isabs(v):
                    d[k] = os.path.normpath(os.path.join(VISO_DIR, v))
        if hasattr(self.processor, 'models_trt_path'):
            for k, v in list(self.processor.models_trt_path.items()):
                if v and not os.path.isabs(v):
                    self.processor.models_trt_path[k] = os.path.normpath(
                        os.path.join(VISO_DIR, v))

    def _ensure_detector(self):
        if not self.processor.models.get('RetinaFace'):
            self.processor.models['RetinaFace'] = self.processor.load_model('RetinaFace')

    def _ensure_arcface(self, swapper_model: str) -> str:
        name = _ARCFACE_MAP.get(swapper_model, 'Inswapper128ArcFace')
        if not self.processor.models.get(name):
            self.processor.models[name] = self.processor.load_model(name)
        if swapper_model == 'Inswapper128':
            self.processor.load_inswapper_iss_emap('Inswapper128')
        return name

    def _get_source_embedding(self, source_img: Image.Image, swapper_model: str):
        src_np = np.array(source_img.convert("RGB"))
        src_t  = torch.from_numpy(src_np).to(self._device).permute(2, 0, 1)
        _, kpss, _ = self.processor.run_detect(src_t, max_num=1)
        if not len(kpss):
            return None, None
        arc = self._ensure_arcface(swapper_model)
        emb, _ = self.processor.run_recognize_direct(src_t, kpss[0],
                                                      similarity_type='Opal',
                                                      arcface_model=arc)
        return emb, src_t

    # ── swap with oval paste-back (no rectangular box) ────────────────────────

    def _swap_frame(self, frame: torch.Tensor, source_emb, swapper_model: str) -> torch.Tensor:
        """
        Detect every face, run the chosen swap model, and paste the result
        back using a smooth oval mask – no rectangular border artifact.
        """
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        bboxes, kpss, _ = self.processor.run_detect(frame)
        if not len(kpss):
            return frame

        result = frame.clone().float()
        dev = self._device

        for kps in kpss:
            try:
                # ── 1. Warp 512×512 aligned face ─────────────────────────────
                tform = trans.SimilarityTransform()
                if swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
                    dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
                    M, _ = faceutil.estimate_norm_arcface_template(kps, src=dst)
                    tform.params[0:2] = M
                elif swapper_model == 'CSCS':
                    tform.estimate(kps, self.processor.FFHQ_kps)
                else:
                    dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
                    tform.estimate(kps, np.squeeze(dst))

                face512 = v2.functional.affine(
                    result.to(torch.uint8),
                    tform.rotation * 57.2958,
                    (tform.translation[0], tform.translation[1]),
                    tform.scale, 0,
                    interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0),
                )
                face512 = v2.functional.crop(face512, 0, 0, 512, 512).float()
                norm = face512 / 255.0   # (3, 512, 512)

                # ── 2. Run swap inference ─────────────────────────────────────
                if swapper_model == 'Inswapper128':
                    inp = t128(norm).permute(1, 2, 0)          # (128,128,3)
                    latent = torch.from_numpy(
                        self.processor.face_swappers.calc_inswapper_latent(source_emb)
                    ).float().to(dev)
                    inp_d = inp.permute(2, 0, 1).unsqueeze(0).contiguous()
                    out   = torch.empty((1, 3, 128, 128), dtype=torch.float32, device=dev)
                    self.processor.face_swappers.run_inswapper(inp_d, latent, out)
                    swap = t512(out.squeeze(0) * 255.0)

                elif swapper_model in ('InStyleSwapper256 Version A',
                                       'InStyleSwapper256 Version B',
                                       'InStyleSwapper256 Version C'):
                    ver  = swapper_model[-1]
                    inp  = t256(norm).permute(1, 2, 0)
                    latent = torch.from_numpy(
                        self.processor.face_swappers.calc_swapper_latent_iss(source_emb, ver)
                    ).float().to(dev)
                    inp_d = inp.permute(2, 0, 1).unsqueeze(0).contiguous()
                    out   = torch.empty((1, 3, 256, 256), dtype=torch.float32, device=dev)
                    self.processor.face_swappers.run_iss_swapper(inp_d, latent, out, ver)
                    swap = t512(out.squeeze(0) * 255.0)

                elif swapper_model == 'SimSwap512':
                    inp  = norm.permute(1, 2, 0)
                    latent = torch.from_numpy(
                        self.processor.face_swappers.calc_swapper_latent_simswap512(source_emb)
                    ).float().to(dev)
                    inp_d = inp.permute(2, 0, 1).unsqueeze(0).contiguous()
                    out   = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=dev)
                    self.processor.face_swappers.run_swapper_simswap512(inp_d, latent, out)
                    swap = out.squeeze(0) * 255.0

                elif swapper_model.startswith('GhostFace'):
                    inp  = t256(norm)
                    inp_d = (inp * 2.0 - 1.0).unsqueeze(0).contiguous()
                    latent = torch.from_numpy(
                        self.processor.face_swappers.calc_swapper_latent_ghost(source_emb)
                    ).float().to(dev)
                    out   = torch.empty((1, 3, 256, 256), dtype=torch.float32, device=dev)
                    self.processor.face_swappers.run_swapper_ghostface(inp_d, latent, out, swapper_model)
                    swap = t512((out.squeeze(0) * 127.5 + 127.5))

                elif swapper_model == 'CSCS':
                    inp  = t256(norm)
                    inp_d = ((inp - 0.5) / 0.5).unsqueeze(0).contiguous()
                    latent = torch.from_numpy(
                        self.processor.face_swappers.calc_swapper_latent_cscs(source_emb)
                    ).float().to(dev)
                    out   = torch.empty((1, 3, 256, 256), dtype=torch.float32, device=dev)
                    self.processor.face_swappers.run_swapper_cscs(inp_d, latent, out)
                    swap = t512((out.squeeze(0) * 0.5 + 0.5) * 255.0)

                else:
                    continue

                # ── 3. Paste back with smooth OVAL mask ───────────────────────
                result = _paste_back(swap, result, tform.inverse, dev, self._oval_512)

            except Exception as e:
                print(f"[VISO] swap error: {e}")

        return result.clamp(0, 255).to(torch.uint8)

    # ── enhancement with oval mask ────────────────────────────────────────────

    def _enhance_face(self, frame: torch.Tensor, restorer_type, restorer_blend, fidelity_weight):
        if restorer_type == 'None':
            return frame

        _, kpss, _ = self.processor.run_detect(frame)
        if not len(kpss):
            return frame

        result = frame.clone().float()
        dev = self._device

        for kps in kpss:
            try:
                tform = trans.SimilarityTransform()
                tform.estimate(kps, self.processor.FFHQ_kps)

                face512 = v2.functional.affine(
                    result.to(torch.uint8),
                    tform.rotation * 57.2958,
                    (tform.translation[0], tform.translation[1]),
                    tform.scale, 0,
                    interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0),
                )
                face512 = v2.functional.crop(face512, 0, 0, 512, 512).float()

                enhanced = self.processor.apply_facerestorer(
                    face512,
                    restorer_det_type='Blend',
                    restorer_type=restorer_type,
                    restorer_blend=restorer_blend,
                    fidelity_weight=fidelity_weight,
                    detect_score=50,
                )

                # Paste with oval mask – no rectangular box
                result = _paste_back(enhanced, result, tform.inverse, dev, self._oval_512)

            except Exception as e:
                print(f"[VISO] enhance error: {e}")

        return result.clamp(0, 255).to(torch.uint8)

    # ── public API ────────────────────────────────────────────────────────────

    def process_image(self, source_img, target_img,
                      swapper_model='Inswapper128',
                      restorer_type='None', restorer_blend=80, fidelity_weight=0.75):
        swapper_model = _SWAPPER_MODEL_MAP.get(swapper_model, 'Inswapper128')
        self._patch_model_paths()
        self._ensure_arcface(swapper_model)

        emb, _ = self._get_source_embedding(source_img, swapper_model)
        if emb is None:
            print("[VISO] No face in source image.")
            return target_img

        tgt = torch.from_numpy(np.array(target_img.convert("RGB"))).to(self._device).permute(2, 0, 1)
        tgt = self._swap_frame(tgt, emb, swapper_model)
        tgt = self._enhance_face(tgt, restorer_type, int(restorer_blend), float(fidelity_weight))

        return Image.fromarray(tgt.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8))

    def process_video(self, source_img, target_video_path, output_path,
                      swapper_model='Inswapper128',
                      restorer_type='None', restorer_blend=80, fidelity_weight=0.75,
                      progress=None):
        swapper_model = _SWAPPER_MODEL_MAP.get(swapper_model, 'Inswapper128')
        self._patch_model_paths()
        self._ensure_arcface(swapper_model)

        emb, _ = self._get_source_embedding(source_img, swapper_model)
        if emb is None:
            raise ValueError("[VISO] No face in source image.")

        cap   = cv2.VideoCapture(target_video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out   = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(rgb).to(self._device).permute(2, 0, 1)
            t   = self._swap_frame(t, emb, swapper_model)
            t   = self._enhance_face(t, restorer_type, int(restorer_blend), float(fidelity_weight))
            np_out = t.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(np_out, cv2.COLOR_RGB2BGR))
            count += 1
            if progress is not None:
                progress(count / max(total, 1), desc=f"Frame {count}/{total}")

        cap.release()
        out.release()
        print(f"[VISO] Done – {count} frames → {output_path}")
        return output_path