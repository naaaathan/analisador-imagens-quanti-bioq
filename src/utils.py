#!/usr/bin/env python3
# utils.py — Utilidades de imagem e conversão (Responsabilidade: funções auxiliares genéricas).

from typing import Optional, Tuple
import cv2
import numpy as np

def bgr_para_gray(img):
    """Converte BGR para escala de cinza. Mantém se já for 1 canal."""
    if img is None:
        raise ValueError("Imagem inválida (None).")
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _resize_keep_aspect(img, target: Tuple[int,int]) -> np.ndarray:
    """Redimensiona mantendo o aspecto, com letterbox (preenchimento em preto) se necessário."""
    h, w = img.shape[:2]
    tw, th = target
    if h == 0 or w == 0:
        return img
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=resized.dtype) if len(resized.shape) == 3 else np.zeros((th, tw), dtype=resized.dtype)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def imagem_para_bytes(img_bgr, resize: Optional[Tuple[int,int]] = None) -> bytes:
    """
    Converte uma imagem BGR do OpenCV para bytes PNG (para exibir no PySimpleGUI).
    Opcionalmente redimensiona (mantendo aspecto).
    """
    if resize is not None:
        # Mantém aspecto para caber na área de visualização
        img_bgr = _resize_keep_aspect(img_bgr, resize)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Falha ao codificar imagem em PNG.")
    return buf.tobytes()
