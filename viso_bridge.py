import os
import sys
import torch
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from typing import Dict, List, Tuple

# Add VisoMaster to path
viso_master_path = os.path.join(os.path.dirname(__file__), "VisoMaster")
if viso_master_path not in sys.path:
    sys.path.insert(0, viso_master_path)

from app.processors.utils import faceutil
from app.processors.models_processor import ModelsProcessor

class MockMainWindow:
    def __init__(self):
        self.model_loading_signal = self.MockSignal()
        self.model_loaded_signal = self.MockSignal()
        self.model_load_dialog = None
        self.parameters = {}
        self.control = {'DetectorModelSelection': 'RetinaFace'}
        self.default_parameters = {'SwapModelSelection': 'Inswapper128'}
        self.target_faces = {}
        self.editFacesButton = self.MockButton()
        self.swapfacesButton = self.MockButton()
        self.faceCompareCheckBox = self.MockButton()
        self.faceMaskCheckBox = self.MockButton()
    class MockSignal:
        def emit(self, *args, **kwargs): pass
    class MockButton:
        def __init__(self): self._checked = False
        def isChecked(self): return self._checked
        def setChecked(self, v): self._checked = v

class VisoBridge:
    def __init__(self, device='cuda'):
        self.mock_win = MockMainWindow()
        self.viso_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "VisoMaster"))
        
        available = ort.get_available_providers()
        if device == 'cuda' and 'CUDAExecutionProvider' not in available:
            device = 'cpu'
        
        self.processor = ModelsProcessor(self.mock_win, device=device)
        self._fix_paths()

        # Sicherer Check für emap (verhindert den 'list' error)
        print("[VISO] Pre-loading models...")
        if not self.processor.models.get('Inswapper128'):
            self.processor.load_model('Inswapper128')
        
        # Wir setzen emap einfach auf ein leeres Array, falls es eine Liste ist
        if isinstance(self.processor.emap, list):
            self.processor.emap = np.array([])

    def _fix_paths(self):
        path_dicts = [self.processor.models_path]
        if hasattr(self.processor, 'models_trt_path'):
            path_dicts.append(self.processor.models_trt_path)
        for d in path_dicts:
            for name in list(d.keys()):
                p = d[name]
                if p and not os.path.isabs(p):
                    rel = p[2:] if p.startswith("./") else p
                    d[name] = os.path.normpath(os.path.join(self.viso_dir, rel))

    def _do_swap_logic(self, target_tensor, source_emb, dev):

        # --- 1. SANITY CHECK: SOURCE EMBEDDING ---
        if source_emb is None:
            print("[VISO] FEHLER: source_emb ist None!")
            return target_tensor
            
        print(f"[VISO] Source Emb Shape: {source_emb.shape}, Min: {source_emb.min():.4f}, Max: {source_emb.max():.4f}")
        
        if np.isnan(source_emb).any() or np.isinf(source_emb).any():
            print("[VISO] KRITISCH: Source Embedding contains NaN or Inf!")
            return target_tensor

        # --- 2. DETECTION ---
        bboxes_t, kpss_5_t, _ = self.processor.run_detect(target_tensor)
        if len(kpss_5_t) == 0: 
            return target_tensor

        res_bgr = target_tensor[[2, 1, 0], :, :].clone().detach().float().contiguous()
        h, w = res_bgr.shape[1], res_bgr.shape[2]

        for i in range(len(kpss_5_t)):
            kps = kpss_5_t[i]
            
            # A. Warp
            aimg, M = faceutil.warp_face_by_face_landmark_5(res_bgr, kps, mode='inswapper')
            
            # B. INPUT-CHECK (Gegen Bild-NaNs)
            if torch.isnan(aimg).any():
                print(f"[VISO] Face {i+1}: Warp delivers NaNs!")
                continue

            # C. INFERENZ VORBEREITUNG
            # Wir probieren den absolut "nackten" Weg ohne komplexe Mathe
            aimg_ready = aimg.clone().to(dev).float()
            # Falls das Modell NaNs wirft, könnte es an zu hohen Werten liegen -> / 255.0
            aimg_ready = aimg_ready / 255.0 
            
            # Embedding auf 512-dim sicherstellen und normalisieren
            latent = source_emb.flatten()
            norm = np.linalg.norm(latent)
            if norm > 0:
                latent = latent / norm
            latent_tensor = torch.from_numpy(latent).to(dev).float().reshape(1, 512)
            
            # D. INFERENZ
            out_buf = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device=dev)
            
            try:
                self.processor.face_swappers.run_inswapper(aimg_ready, latent_tensor, out_buf)
            except Exception as e:
                print(f"[VISO] Crash during run_inswapper: {e}")
                continue

            if torch.isnan(out_buf).any():
                print(f"[VISO] Face {i+1}: Inferenz delivers NaN (despite 0-1 Norm).")
                # Letzter verzweifelter Versuch: Komplett ohne Normalisierung
                out_buf.fill_(0)
                self.processor.face_swappers.run_inswapper(aimg.to(dev).float(), latent_tensor, out_buf)

            if torch.isnan(out_buf).any():
                continue

            # E. PASTE BACK
            out_face = out_buf[0].clone().detach()
            if out_face.max() <= 1.05:
                out_face = out_face * 255.0
            out_face = out_face.clamp(0, 255).contiguous()

            # Maske (Originalgröße)
            mask_128 = np.zeros((128, 128), dtype=np.float32)
            cv2.rectangle(mask_128, (10, 10), (118, 118), (1.0), -1)
            mask_128 = cv2.GaussianBlur(mask_128, (15, 15), 0)
            
            M_c2o = cv2.invertAffineTransform(M)
            full_mask = cv2.warpAffine(mask_128, M_c2o, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            full_mask_t = torch.from_numpy(full_mask).to(dev).float().unsqueeze(0).contiguous()

            res_bgr = faceutil.paste_back(out_face, M_c2o, res_bgr, full_mask_t)
            print(f"[VISO] Face {i+1} successful.")

        return res_bgr[[2, 1, 0], :, :].contiguous()

    def process_image(self, source_img: Image.Image, target_img: Image.Image, swapper_model="Inswapper128") -> Image.Image:
        self._fix_paths()
        dev = self.processor.device
        
        # Source Embedding
        source_np = np.array(source_img.convert("RGB"))
        source_tensor = torch.from_numpy(source_np).to(dev).permute(2,0,1)
        _, kps_s, _ = self.processor.run_detect(source_tensor, max_num=1)
        if len(kps_s) == 0: 
            print("[VISO] No face detected in source image.")
            return target_img
        source_emb, _ = self.processor.run_recognize_direct(source_tensor, kps_s[0])

        # Target Verarbeitung
        target_np = np.array(target_img.convert("RGB"))
        target_tensor = torch.from_numpy(target_np).to(dev).permute(2,0,1)
        
        result_tensor = self._do_swap_logic(target_tensor, source_emb, dev)
        
        result_np = result_tensor.permute(1,2,0).cpu().numpy().clip(0, 255).astype(np.uint8)
        return Image.fromarray(result_np)

    def process_video(self, source_img: Image.Image, target_video_path: str, output_path: str, swapper_model="Inswapper128", progress=None):
        self._fix_paths()
        dev = self.processor.device

        # Source Embedding
        source_np = np.array(source_img.convert("RGB"))
        source_tensor = torch.from_numpy(source_np).to(dev).permute(2,0,1)
        _, kps_s, _ = self.processor.run_detect(source_tensor, max_num=1)
        if len(kps_s) == 0: 
            raise ValueError("No face detected in source image")
        source_emb, _ = self.processor.run_recognize_direct(source_tensor, kps_s[0])

        cap = cv2.VideoCapture(target_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).to(dev).permute(2,0,1)
            
            # Hier nutzen wir jetzt die neue, funktionierende Logik
            result_tensor = self._do_swap_logic(frame_tensor, source_emb, dev)
            
            res_np = result_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)
            out.write(cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR))
            
            count += 1
            if progress: progress(count/total, desc=f"Frame {count}/{total}")

        cap.release()
        out.release()
        return output_path