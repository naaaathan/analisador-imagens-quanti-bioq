#!/usr/bin/env python3
"""
roi_selector.py — Responsabilidade: detecção/seleção de ROIs (poços e fundos) e cálculos associados.

Funcionalidades:
- HoughCircles para detectar poços (círculos) com limites de raio plausíveis.
- Pontuação por contraste (anel − disco) para escolher melhores poços.
- Seleção manual do poço e do(s) fundo(s) via cv2.selectROI.
- Cálculo da média do poço e do fundo (anel, ROI única ou múltiplas ROIs).
- Redetecção guiada por fundo(s) com limiar adaptativo + morfologia + deduplicação + limitação de raio.
- Filtros pós-detecção:
    • fotométrico: descarte de discos parecidos com o fundo
    • cromático (Lab/HSV): mantém apenas poços com “assinatura” de líquido
- Desenho de overlays para visualizar poços e fundos.

Requisitos cobertos:
- Seleção de ROI de amostra e ROI(s) de fundo (inclusive múltiplas).
- Correção de fundo no cálculo da intensidade (fundo - poço).
- Modo de calibração e de análise compartilham utilitários consistentes.
"""
from typing import Optional, Tuple, List
import math
import cv2
import numpy as np
from utils import bgr_para_gray

__all__ = [
    "detectar_pocos",
    "pontuar_pocos",
    "escolher_melhor_poco",
    "detectar_circulos",
    "selecionar_poco_manual_cv",
    "selecionar_roi_fundo_cv",
    "calcular_medias_roi",
    "calcular_medias_roi_multi_bg",
    "desenhar_overlay",
    "desenhar_overlays_multiplos",
    "filtrar_pocos_por_fundo",
    "filtrar_pocos_por_fundos",
    "redetectar_pocos_guiado_por_fundo",
    "redetectar_pocos_guiado_por_fundos",
    "deduplicar_circulos",
    "limitar_raio_por_referencia",
    "filtrar_pocos_por_cor",
]


def _get_screen_bounds(default=(1600, 900)):
    """Tenta descobrir o tamanho da tela (fallback para 1600x900)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        # Usamos 92% da tela para dar folga às bordas/janelas do SO
        return int(sw * 0.92), int(sh * 0.92)
    except Exception:
        return default

def _fit_for_display(img_bgr, max_w, max_h):
    """
    Redimensiona a imagem para caber na tela mantendo o aspect ratio.
    Retorna (imagem_exibida, escala_aplicada).
    """
    h, w = img_bgr.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1.0:
        disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = img_bgr.copy()
    return disp, scale


# ========= DETECÇÃO E PONTUAÇÃO DE POÇOS =========

def detectar_pocos(imagem,
                   dp: float = 1.2,
                   min_dist: Optional[int] = None,
                   param1: int = 80,
                   param2: int = 25,
                   min_radius: Optional[int] = None,
                   max_radius: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """Detecta possíveis poços (círculos) via HoughCircles e retorna [(x, y, r), ...]."""
    gray = bgr_para_gray(imagem)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    if min_dist is None:
        min_dist = max(20, min(h, w) // 18)
    if min_radius is None:
        min_radius = max(8, int(min(h, w) * 0.02))
    if max_radius is None:
        max_radius = int(min(h, w) * 0.12)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp, min_dist,
        param1=param1, param2=param2,
        minRadius=int(min_radius), maxRadius=int(max_radius)
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(int(x), int(y), int(r)) for (x, y, r) in circles]

def _score_poco(gray: np.ndarray, c: Tuple[int, int, int]) -> float:
    """Score por contraste (anel externo − disco interno)."""
    x, y, r = c
    mask_disc = np.zeros_like(gray, np.uint8)
    cv2.circle(mask_disc, (x, y), r, 255, -1)
    r2 = int(r * 1.35)
    r2 = min(r2, x, y, gray.shape[1] - x, gray.shape[0] - y)
    mask_ring = np.zeros_like(gray, np.uint8)
    cv2.circle(mask_ring, (x, y), r2, 255, -1)
    cv2.circle(mask_ring, (x, y), r, 0, -1)
    mean_disc = cv2.mean(gray, mask_disc)[0]   # type: ignore
    mean_ring = cv2.mean(gray, mask_ring)[0]   # type: ignore
    return float(mean_ring - mean_disc)

def pontuar_pocos(imagem, circles: List[Tuple[int, int, int]]) -> List[float]:
    gray = bgr_para_gray(imagem)
    return [_score_poco(gray, c) for c in circles]

def escolher_melhor_poco(imagem, circles: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
    if not circles:
        return None
    scores = pontuar_pocos(imagem, circles)
    best_i = int(np.argmax(scores))
    return circles[best_i]

def detectar_circulos(imagem) -> Optional[Tuple[int, int, int]]:
    cs = detectar_pocos(imagem)
    return escolher_melhor_poco(imagem, cs)

# ========= SELEÇÕES MANUAIS =========

def selecionar_poco_manual_cv(imagem) -> Optional[Tuple[int, int, int]]:
    """Usuário desenha retângulo; convertemos para círculo (centro e raio = min(w,h)/2)."""
    titulo = "Selecione o POÇO (ENTER confirma, ESC cancela)"
    rect = cv2.selectROI(titulo, imagem, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(titulo)
    if rect == (0, 0, 0, 0):
        return None
    x, y, w, h = map(int, rect)
    cx = x + w // 2
    cy = y + h // 2
    r = max(5, int(min(w, h) // 2))
    return (cx, cy, r)

def selecionar_roi_fundo_cv(img_bgr):
    """
    Abre uma janela para selecionar a ROI de fundo, sempre ajustando a imagem
    para caber na tela. Mapeia a ROI de volta para a resolução original.
    ENTER/SPACE confirma, 'c' cancela.
    Retorna (x, y, w, h) na resolução original ou None se cancelado.
    """
    win = "Selecione ROI de Fundo (ENTER/SPACE confirma, 'c' cancela)"
    max_w, max_h = _get_screen_bounds()
    disp, scale = _fit_for_display(img_bgr, max_w, max_h)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
    # showCrosshair ajuda a apontar; fromCenter=False = arrastar canto a canto
    rect = cv2.selectROI(win, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    if rect is None or rect == (0, 0, 0, 0):
        return None

    x, y, w, h = rect
    if scale != 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        w = int(round(w / scale))
        h = int(round(h / scale))

    # Clamping para garantir que a ROI fique dentro da imagem original
    H, W = img_bgr.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return (x, y, w, h)

# ========= CÁLCULO DE MÉDIAS =========

def _media_mascara(gray, mask) -> float:
    return float(cv2.mean(gray, mask=mask)[0])  # type: ignore

def calcular_medias_roi(imagem,
                        circulo: Tuple[int, int, int],
                        bg_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, float, float]:
    """Calcula média do poço e do fundo (anel ou ROI retangular). Retorna (poço, fundo, fundo−poço)."""
    x, y, r = circulo
    gray = bgr_para_gray(imagem)
    mask_poco = np.zeros_like(gray, np.uint8)
    cv2.circle(mask_poco, (x, y), r, 255, -1)
    media_poco = _media_mascara(gray, mask_poco)
    if bg_rect is None:
        r_fundo = int(r * 1.5)
        r_fundo = min(r_fundo, x, y, gray.shape[1] - x, gray.shape[0] - y)
        mask_fundo = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_fundo, (x, y), r_fundo, 255, -1)
        cv2.circle(mask_fundo, (x, y), r, 0, -1)
        media_fundo = _media_mascara(gray, mask_fundo)
    else:
        bx, by, bw, bh = bg_rect
        roi_bg = gray[by:by + bh, bx:bx + bw]
        media_fundo = float(np.mean(roi_bg)) if roi_bg.size > 0 else media_poco
    return media_poco, media_fundo, (media_fundo - media_poco)

def calcular_medias_roi_multi_bg(imagem,
                                 circulo: Tuple[int, int, int],
                                 bg_rects: List[Tuple[int, int, int, int]]) -> Tuple[float, float, float]:
    """Versão com várias ROIs de fundo: usa a média das médias das ROIs de fundo."""
    x, y, r = circulo
    gray = bgr_para_gray(imagem)
    mask_poco = np.zeros_like(gray, np.uint8)
    cv2.circle(mask_poco, (x, y), r, 255, -1)
    media_poco = _media_mascara(gray, mask_poco)
    if not bg_rects:
        r_fundo = int(r * 1.5)
        r_fundo = min(r_fundo, x, y, gray.shape[1] - x, gray.shape[0] - y)
        mask_fundo = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_fundo, (x, y), r_fundo, 255, -1)
        cv2.circle(mask_fundo, (x, y), r, 0, -1)
        media_fundo = _media_mascara(gray, mask_fundo)
    else:
        medias = []
        for bx, by, bw, bh in bg_rects:
            roi_bg = gray[by:by + bh, bx:bx + bw]
            if roi_bg.size > 0:
                medias.append(float(np.mean(roi_bg)))
        media_fundo = float(np.mean(medias)) if medias else media_poco
    return media_poco, media_fundo, (media_fundo - media_poco)

# ========= FILTROS FOTOMÉTRICOS =========

def filtrar_pocos_por_fundo(
    imagem,
    circles: List[Tuple[int, int, int]],
    bg_rect: Tuple[int, int, int, int],
    abs_diff_thresh: float = 6.0,
    min_contrast_delta: float = 3.0,
) -> List[Tuple[int, int, int]]:
    """Elimina poços cujo disco é 'parecido' com o fundo (diferença baixa e pouco contraste anel−disco)."""
    if not circles or bg_rect is None:
        return circles
    gray = bgr_para_gray(imagem)
    bx, by, bw, bh = map(int, bg_rect)
    roi_bg = gray[by:by + bh, bx:bx + bw]
    if roi_bg.size == 0:
        return circles
    mean_bg = float(np.mean(roi_bg))
    kept = []
    for (x, y, r) in circles:
        mask_disc = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_disc, (x, y), r, 255, -1)
        mean_disc = cv2.mean(gray, mask_disc)[0]  # type: ignore
        r2 = int(r * 1.35)
        r2 = min(r2, x, y, gray.shape[1] - x, gray.shape[0] - y)
        mask_ring = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_ring, (x, y), r2, 255, -1)
        cv2.circle(mask_ring, (x, y), r, 0, -1)
        mean_ring = cv2.mean(gray, mask_ring)[0]  # type: ignore
        delta_contraste = float(mean_ring - mean_disc)
        diff_bg = abs(float(mean_disc) - mean_bg)
        if (diff_bg >= abs_diff_thresh) or (delta_contraste >= min_contrast_delta):
            kept.append((int(x), int(y), int(r)))
    return kept

def filtrar_pocos_por_fundos(
    imagem,
    circles: List[Tuple[int, int, int]],
    bg_rects: List[Tuple[int, int, int, int]],
    abs_diff_thresh: float = 6.0,
    min_contrast_delta: float = 3.0,
) -> List[Tuple[int, int, int]]:
    """Versão multi-ROIs: descarta se for parecido com QUALQUER fundo e com pouco contraste anel−disco."""
    if not circles or not bg_rects:
        return circles
    gray = bgr_para_gray(imagem)
    means_bg = []
    for bx, by, bw, bh in bg_rects:
        roi_bg = gray[by:by + bh, bx:bx + bw]
        if roi_bg.size > 0:
            means_bg.append(float(np.mean(roi_bg)))
    if not means_bg:
        return circles
    kept = []
    for (x, y, r) in circles:
        mask_disc = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_disc, (x, y), r, 255, -1)
        mean_disc = cv2.mean(gray, mask_disc)[0]  # type: ignore
        r2 = int(r * 1.35)
        r2 = min(r2, x, y, gray.shape[1] - x, gray.shape[0] - y)
        mask_ring = np.zeros_like(gray, np.uint8)
        cv2.circle(mask_ring, (x, y), r2, 255, -1)
        cv2.circle(mask_ring, (x, y), r, 0, -1)
        mean_ring = cv2.mean(gray, mask_ring)[0]  # type: ignore
        delta_contraste = float(mean_ring - mean_disc)
        similar_a_algum_bg = any(abs(float(mean_disc) - m_bg) < abs_diff_thresh and delta_contraste < min_contrast_delta
                                 for m_bg in means_bg)
        if not similar_a_algum_bg:
            kept.append((int(x), int(y), int(r)))
    return kept

# ========= AUXILIARES DE REDETECÇÃO =========

def _bg_stats(gray: np.ndarray, bg_rects: List[Tuple[int,int,int,int]]) -> tuple[list[float], list[float], float]:
    """Calcula médias e desvios nas ROIs de fundo e devolve limiar adaptativo T baseado no ruído."""
    means, stds = [], []
    for bx, by, bw, bh in bg_rects:
        roi = gray[by:by+bh, bx:bx+bw]
        if roi.size > 0:
            means.append(float(np.mean(roi)))
            stds.append(float(np.std(roi)))
    if not means:
        return [], [], 8.0
    T = max(8.0, 2.0 * (float(np.median(stds)) if stds else 4.0))
    return means, stds, T

def deduplicar_circulos(circulos: List[Tuple[int,int,int]], prox_ratio: float = 0.9) -> List[Tuple[int,int,int]]:
    """Remove duplicatas muito próximas; mantém o círculo maior."""
    if not circulos:
        return []
    kept: List[Tuple[int,int,int]] = []
    for x, y, r in sorted(circulos, key=lambda c: c[2], reverse=True):
        ok = True
        for xk, yk, rk in kept:
            d = ((x-xk)**2 + (y-yk)**2) ** 0.5
            if d < prox_ratio * min(r, rk):
                ok = False
                break
        if ok:
            kept.append((int(x), int(y), int(r)))
    return kept

def limitar_raio_por_referencia(min_r: Optional[int], max_r: Optional[int],
                                ref_circles: Optional[List[Tuple[int,int,int]]],
                                img_shape: Tuple[int,int]) -> Tuple[int,int]:
    """Se houver referência, usa [0.75*r_med, 1.35*r_med]; caso contrário, heurística por tamanho."""
    h, w = img_shape
    if ref_circles:
        r_med = int(np.median([r for (_,_,r) in ref_circles]))
        min_r_new = max(8, int(0.75 * r_med))
        max_r_new = max(min_r_new + 2, int(1.35 * r_med))
    else:
        min_r_new = max(8, int(min(h, w) * 0.02)) if min_r is None else min_r
        max_r_new = int(min(h, w) * 0.12) if max_r is None else max_r
    return int(min_r_new), int(max_r_new)

# ========= REDETECÇÃO GUIADA POR FUNDO(S) =========

def redetectar_pocos_guiado_por_fundo(
    imagem,
    bg_rect: Tuple[int, int, int, int],
    abs_diff_thresh: int = 6,
    dp: float = 1.2,
    min_dist: Optional[int] = None,
    param1: int = 80,
    param2: int = 35,
    min_radius: Optional[int] = None,
    max_radius: Optional[int] = None,
    ref_circles: Optional[List[Tuple[int,int,int]]] = None,
) -> List[Tuple[int, int, int]]:
    """Versão simples (1 ROI de fundo). Mantida por compatibilidade."""
    return redetectar_pocos_guiado_por_fundos(
        imagem, [bg_rect], abs_diff_thresh, dp, min_dist, param1, param2, min_radius, max_radius, ref_circles
    )

def redetectar_pocos_guiado_por_fundos(
    imagem,
    bg_rects: List[Tuple[int, int, int, int]],
    abs_diff_thresh: Optional[int] = None,
    dp: float = 1.2,
    min_dist: Optional[int] = None,
    param1: int = 80,
    param2: int = 35,
    min_radius: Optional[int] = None,
    max_radius: Optional[int] = None,
    ref_circles: Optional[List[Tuple[int,int,int]]] = None,
) -> List[Tuple[int, int, int]]:
    """
    Redetecta com Hough após suprimir pixels próximos a QUALQUER nível médio de fundo.
    Usa limiar adaptativo baseado no ruído das ROIs de fundo; aplica morfologia, blur, limitação de raio e deduplicação.
    """
    gray = bgr_para_gray(imagem)
    h, w = gray.shape[:2]
    if not bg_rects:
        return []
    means, stds, T_adapt = _bg_stats(gray, bg_rects)
    if not means:
        return []
    T = int(abs_diff_thresh if abs_diff_thresh is not None else round(T_adapt))
    # mantém apenas pixels cuja diferença para TODAS as médias de fundo seja > T
    min_diff = None
    for m in means:
        d = cv2.absdiff(gray, np.full_like(gray, int(round(m))))
        min_diff = d if min_diff is None else np.minimum(min_diff, d)
    keep_mask = (min_diff > T) #type: ignore
    thr = gray.copy()
    thr[~keep_mask] = 0
    # ruído/bordas finas
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    thr = cv2.GaussianBlur(thr, (5, 5), 0)
    # faixa de raios e minDist coerentes
    min_r, max_r = limitar_raio_por_referencia(min_radius, max_radius, ref_circles, (h, w))
    if min_dist is None:
        est_r = int(np.median([r for (_,_,r) in ref_circles])) if ref_circles else int((min_r + max_r) / 2)
        min_dist = max(20, int(1.2 * est_r))
    circles = cv2.HoughCircles(
        thr, cv2.HOUGH_GRADIENT, dp, min_dist,
        param1=param1, param2=param2,
        minRadius=int(min_r), maxRadius=int(max_r)
    )
    if circles is None:
        return []
    cs = [(int(x), int(y), int(r)) for (x, y, r) in np.round(circles[0, :]).astype(int)]
    cs = deduplicar_circulos(cs, prox_ratio=0.9)
    cs = filtrar_pocos_por_fundos(imagem, cs, bg_rects, abs_diff_thresh=float(T), min_contrast_delta=3.0)
    return cs

# ========= FILTRO CROMÁTICO (Lab/HSV) =========

def _mean_lab_hsv(im_bgr, mask):
    """Média Lab e HSV dentro da máscara binária (0/255)."""
    lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    m_lab = [cv2.mean(lab[:,:,i], mask=mask)[0] for i in range(3)]  # type: ignore
    m_hsv = [cv2.mean(hsv[:,:,i], mask=mask)[0] for i in range(3)]  # type: ignore
    return (m_lab, m_hsv)

def _deltaE76(Lab1, Lab2):
    dL = Lab1[0]-Lab2[0]; da = Lab1[1]-Lab2[1]; db = Lab1[2]-Lab2[2]
    return math.sqrt(dL*dL + da*da + db*db)

def _bg_means_lab_hsv(im_bgr, bg_rects):
    """Média global (sobre todas as ROIs de fundo) em Lab e HSV."""
    lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    acc_lab, acc_hsv, n = np.zeros(3), np.zeros(3), 0
    for bx, by, bw, bh in bg_rects:
        roi_lab = lab[by:by+bh, bx:bx+bw]
        roi_hsv = hsv[by:by+bh, bx:bx+bw]
        if roi_lab.size == 0:
            continue
        acc_lab += np.array([np.mean(roi_lab[:,:,i]) for i in range(3)])
        acc_hsv += np.array([np.mean(roi_hsv[:,:,i]) for i in range(3)])
        n += 1
    if n == 0:
        return [0,0,0], [0,0,0]
    return (acc_lab/n).tolist(), (acc_hsv/n).tolist()

def filtrar_pocos_por_cor(
    imagem_bgr,
    circles: List[Tuple[int,int,int]],
    bg_rects: List[Tuple[int,int,int,int]],
    deltaE_min: float = 8.0,
    s_min: float = 15.0,     # Saturação mínima (HSV) do disco
    v_diff_min: float = 8.0  # Diferença mínima de V entre disco e fundo
) -> List[Tuple[int,int,int]]:
    """
    Mantém poços cujo DISCO tem cor compatível com 'líquido' em relação ao fundo:
      - ΔE(Lab_disc, Lab_bg) >= deltaE_min   OU
      - (S_disc >= s_min) E (|V_disc - V_bg| >= v_diff_min)
    """
    if not circles or not bg_rects:
        return circles
    lab_bg, hsv_bg = _bg_means_lab_hsv(imagem_bgr, bg_rects)
    kept = []
    h, w = imagem_bgr.shape[:2]
    for (x, y, r) in circles:
        mask_disc = np.zeros((h, w), np.uint8)
        cv2.circle(mask_disc, (x, y), int(r), 255, -1)
        (lab_disc, hsv_disc) = _mean_lab_hsv(imagem_bgr, mask_disc)
        dE = _deltaE76(lab_disc, lab_bg)
        Sd, Vd = hsv_disc[1], hsv_disc[2]
        Vbg = hsv_bg[2]
        cond_lab = dE >= deltaE_min
        cond_hsv = (Sd >= s_min) and (abs(Vd - Vbg) >= v_diff_min)
        if cond_lab or cond_hsv:
            kept.append((int(x), int(y), int(r)))
    return kept

# ========= DESENHO =========

def desenhar_overlay(imagem,
                     circulo: Tuple[int, int, int],
                     bg_rect: Optional[Tuple[int, int, int, int]] = None):
    """Desenha o círculo selecionado (vermelho) e o fundo (retângulo verde) ou anel."""
    annotated = imagem.copy()
    x, y, r = circulo
    cv2.circle(annotated, (x, y), r, (0, 0, 255), 2)
    if bg_rect is not None:
        bx, by, bw, bh = bg_rect
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    else:
        r_fundo = int(r * 1.5)
        r_fundo = min(r_fundo, x, y, annotated.shape[1] - x, annotated.shape[0] - y)
        cv2.circle(annotated, (x, y), r_fundo, (0, 255, 0), 2)
    return annotated

def desenhar_overlays_multiplos(imagem,
                                circles: List[Tuple[int, int, int]],
                                selected_index: int,
                                bg_rect=None):
    """
    Desenha:
      - poço selecionado: vermelho grosso
      - outros poços: amarelo fino
      - fundos: todos os retângulos verdes (se lista), um retângulo (se tupla) ou anel verde (se None)
    """
    annotated = imagem.copy()
    # outros (amarelo)
    for i, (x, y, r) in enumerate(circles):
        if i == selected_index:
            continue
        color = (0, 255, 255)
        cv2.circle(annotated, (x, y), r, color, 1)
        cv2.putText(annotated, f"{i+1}", (x - r, y - r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # selecionado (vermelho)
    if circles:
        sx, sy, sr = circles[selected_index]
        cv2.circle(annotated, (sx, sy), sr, (0, 0, 255), 2)
        cv2.putText(annotated, f"{selected_index+1}", (sx - sr, sy - sr - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    # fundos
    if isinstance(bg_rect, list):
        for (bx, by, bw, bh) in bg_rect:
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    elif bg_rect is not None:
        bx, by, bw, bh = bg_rect
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    else:
        if circles:
            r_fundo = int(sr * 1.5) #type: ignore
            r_fundo = min(r_fundo, sx, sy, annotated.shape[1] - sx, annotated.shape[0] - sy) #type: ignore
            cv2.circle(annotated, (sx, sy), r_fundo, (0, 255, 0), 2) #type: ignore
    return annotated
