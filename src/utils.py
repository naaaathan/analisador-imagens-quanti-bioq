#!/usr/bin/env python3
# utils.py — Utilitários adicionais: extração de features por espaço de cor, compressão e mapeamento aproximado RGB→nm.
from typing import Optional, Tuple, List, Iterable
import cv2
import numpy as np
import os
import math

def bgr_para_gray(img):
    if img is None:
        raise ValueError("Imagem inválida (None).")
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _resize_keep_aspect(img, target: Tuple[int,int]) -> np.ndarray:
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
    if resize is not None:
        img_bgr = _resize_keep_aspect(img_bgr, resize)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Falha ao codificar imagem em PNG.")
    return buf.tobytes()

# -------------------- Novos utilitários --------------------

def mean_color_in_circle_bgr(img_bgr, circle: Tuple[int,int,int]):
    """Retorna média B,G,R dentro do círculo (disco)."""
    x,y,r = circle
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(mask, (x,y), r, 255, -1)
    means = cv2.mean(img_bgr, mask=mask)[:3]  # B,G,R #type: ignore
    # converter para R,G,B
    return [float(means[2]), float(means[1]), float(means[0])]

def extract_features(img_bgr, circle: Tuple[int,int,int], color_space: str = "RGB", components: Iterable[str] = ("ALL",)):
    """
    Extrai features médias dentro do disco para o espaço de cor e componentes selecionados.
    color_space: "RGB", "HSV", "LAB"
    components: tuple/list com elementos como "H","S","V","L","a","b" ou "ALL" para todos os canais.
    Retorna lista de floats (1..3 elementos) conforme seleção.
    """
    rgb = mean_color_in_circle_bgr(img_bgr, circle)  # R,G,B
    r,g,b = rgb
    if color_space.upper() == "RGB":
        vals = {"R": r, "G": g, "B": b}
    elif color_space.upper() == "HSV":
        hsv = cv2.cvtColor(np.uint8([[ [b,g,r] ]]), cv2.COLOR_BGR2HSV)[0,0]  # H,S,V #type: ignore
        vals = {"H": float(hsv[0]), "S": float(hsv[1]), "V": float(hsv[2])}
    elif color_space.upper() in ("LAB", "L*a*b*", "L\*a\*b*"):
        lab = cv2.cvtColor(np.uint8([[ [b,g,r] ]]), cv2.COLOR_BGR2LAB)[0,0]  # L,a,b #type: ignore
        vals = {"L": float(lab[0]), "a": float(lab[1]), "b": float(lab[2])}
    else:
        raise ValueError(f"Espaço de cor desconhecido: {color_space}")
    # processamento de componentes
    if "ALL" in components:
        # preservar uma ordem sensata dependendo do espaço de cor
        if color_space.upper() == "RGB":
            return [vals["R"], vals["G"], vals["B"]]
        elif color_space.upper() == "HSV":
            return [vals["H"], vals["S"], vals["V"]]
        else:
            return [vals["L"], vals["a"], vals["b"]]
    else:
        out = []
        for c in components:
            if c not in vals:
                raise ValueError(f"Componente inválido para {color_space}: {c}")
            out.append(vals[c])
        return out

def save_compressed_versions(src_folder: str, dst_folder: str, quality_levels: List[int] = [95,75,50,25]):
    """
    Cria subpastas jpeg_{quality} dentro de dst_folder e salva versões JPEG com a qualidade dada.
    Retorna lista de caminhos onde as pastas foram criadas.
    """
    if not os.path.isdir(src_folder):
        raise ValueError("src_folder inexistente")
    os.makedirs(dst_folder, exist_ok=True)
    created = []
    for q in quality_levels:
        sub = os.path.join(dst_folder, f"jpeg_{q}")
        os.makedirs(sub, exist_ok=True)
        created.append(sub)
    for fn in sorted(os.listdir(src_folder)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        full = os.path.join(src_folder, fn)
        img = cv2.imread(full, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        for q in quality_levels:
            sub = os.path.join(dst_folder, f"jpeg_{q}")
            outp = os.path.join(sub, fn.rsplit(".",1)[0] + ".jpg")
            # cv2.imwrite respeita o parâmetro de qualidade para JPG
            cv2.imwrite(outp, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return created

# Mapeamento aproximado RGB -> comprimento de onda (nm).
# NOTA: Não há bijeção direta; esta função usa um mapeamento heurístico via matiz HSV -> comprimento de onda.
# Matiz no OpenCV é [0,179] correspondendo aproximadamente a vermelho→amarelo→verde→ciano→azul→magenta→vermelho.
# Vamos mapear a matiz para a faixa visível 380..780 nm aproximadamente (invertido quando necessário).
def approx_wavelength_from_rgb(r,g,b):
    # converter cor única para HSV usando OpenCV (entrada BGR)
    hsv = cv2.cvtColor(np.uint8([[[b,g,r]]]), cv2.COLOR_BGR2HSV)[0,0] #type: ignore
    h = float(hsv[0])  # 0..179
    # Mapear 0..179 -> 380..780 (wrap-around no vermelho); esta é uma aproximação grosseira
    nm = 380.0 + (h / 179.0) * 400.0
    return float(nm)

def spectral_curve_from_roi(img_bgr, circle: Tuple[int,int,int], n_samples: int = 100):
    """
    Gera um 'gráfico espectral' aproximado para a ROI: converte cada pixel RGB para uma 'wavelength' via hue
    e acumula intensidade por faixa. Retorna (wavelengths, intensities).
    Esta é uma heuristic visualization — suficiente para o projeto didático.
    """
    x,y,r = circle
    h_img, w_img = img_bgr.shape[:2]
    mask = np.zeros((h_img,w_img), dtype=np.uint8)
    cv2.circle(mask, (x,y), r, 255, -1)
    roi = img_bgr.copy()
    roi[mask==0] = 0
    # obter pixels dentro da máscara
    pixels = roi[mask==255]
    if pixels.size == 0:
        return [], []
    # calcular comprimento de onda para cada pixel via approx_wavelength_from_rgb
    nms = [approx_wavelength_from_rgb(int(px[2]), int(px[1]), int(px[0])) for px in pixels]
    vals = [int(0.299*px[2] + 0.587*px[1] + 0.114*px[0]) for px in pixels]  # aproximação da luminância
    # agrupar em n_samples entre 380..780
    bins = np.linspace(380, 780, n_samples+1)
    inds = np.digitize(nms, bins) - 1
    intens = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=int)
    for idx, v in zip(inds, vals):
        if 0 <= idx < n_samples:
            intens[idx] += v
            counts[idx] += 1
    # intensidade média por grupo
    with np.errstate(divide='ignore', invalid='ignore'):
        avg = np.divide(intens, np.maximum(1, counts))
    centers = 0.5*(bins[:-1] + bins[1:])
    return centers.tolist(), avg.tolist()
