#!/usr/bin/env python3
# gui.py — Interface (PySimpleGUI) + orquestração do fluxo (Calibração e Análise).

from typing import Optional, Any, Tuple, List, Dict
import os
import PySimpleGUI as sg

from image_loader import carregar_imagem
from roi_selector import (
    detectar_pocos,
    pontuar_pocos,
    selecionar_roi_fundo_cv,
    calcular_medias_roi,
    calcular_medias_roi_multi_bg,
    desenhar_overlays_multiplos,
    filtrar_pocos_por_fundos,
    redetectar_pocos_guiado_por_fundos,
    filtrar_pocos_por_cor,
)
from regression_model import ModeloCalibracao
from utils import imagem_para_bytes

# --- Config de pré-visualização (LxA em pixels) ---
PREVIEW_SIZE = (720, 540)

# Modelo (compartilhado entre os modos)
modelo_global = ModeloCalibracao()

# Contexto compartilhado entre Calibração e Análise (para refletir os refinamentos)
contexto_compartilhado = {
    "bg_rects": [],        # list[tuple(x,y,w,h)] — ROIs de fundo da calibração
    "ref_circles": [],     # list[tuple(x,y,r)]   — poços filtrados na calibração
    "last_img_shape": None # (h, w) da última imagem de calibração
}

# ---------- Helpers de UI ----------

def _formatar_labels_pocos(circles: List[Tuple[int,int,int]], scores: List[float]) -> List[str]:
    labels = []
    for i, ((_, _, r), s) in enumerate(zip(circles, scores)):
        labels.append(f"#{i+1}  r={r}  score={s:.1f}")
    return labels

def _index_por_label(label: str) -> int:
    try:
        num = label.split()[0].strip("#")
        return max(0, int(num) - 1)
    except Exception:
        return 0

def _fmt_bg_list(bg_rects: List[Tuple[int,int,int,int]]) -> List[str]:
    return [f"{i+1}: x={bx},y={by},w={bw},h={bh}" for i,(bx,by,bw,bh) in enumerate(bg_rects)]

# ========================= CALIBRAÇÃO =========================

def iniciar_interface():
    """Loop que permite alternar Calibração ↔ Análise mantendo o mesmo modelo/calibração."""
    while True:
        wants_analysis = run_calibration_window(modelo_global)
        if wants_analysis is None:  # usuário fechou
            break
        if wants_analysis:
            wants_back = run_analysis_window(modelo_global)
            if wants_back is None:
                break

def run_calibration_window(modelo: ModeloCalibracao) -> Optional[bool]:
    sg.theme("LightBlue")

    # -------- Painel Avançado (Calibração) --------
    adv_frame_calib = sg.Frame(
        "Avançado",
        [
            [sg.Checkbox("Redetectar guiado pelo fundo", key="-ADV-REDET-", default=True),
             sg.Checkbox("Filtro fotométrico (fundo/contraste)", key="-ADV-PHOTO-", default=True),
             sg.Checkbox("Filtro de cor (Lab/HSV)", key="-ADV-COLOR-", default=True)],
            [sg.Text("param2 (Hough):", size=(16,1)),
             sg.Slider(range=(10,80), default_value=35, resolution=1, orientation="h", size=(30,15), key="-ADV-PARAM2-")],
            [sg.Text("Δabs (disco−fundo):", size=(16,1)),
             sg.Slider(range=(0,25), default_value=6, resolution=1, orientation="h", size=(30,15), key="-ADV-ABS-"),
             sg.Text("Δcontraste (anel−disco):", size=(22,1)),
             sg.Slider(range=(0,25), default_value=3, resolution=1, orientation="h", size=(30,15), key="-ADV-CONTR-")],
            [sg.Text("ΔE_min (Lab):", size=(16,1)),
             sg.Slider(range=(0,25), default_value=8, resolution=1, orientation="h", size=(30,15), key="-ADV-DE-"),
             sg.Text("S_min (HSV):", size=(12,1)),
             sg.Slider(range=(0,255), default_value=15, resolution=1, orientation="h", size=(30,15), key="-ADV-SMIN-"),
             sg.Text("|V−Vfundo|:", size=(10,1)),
             sg.Slider(range=(0,50), default_value=8, resolution=1, orientation="h", size=(30,15), key="-ADV-VDIFF-")],
        ],
        relief=sg.RELIEF_SUNKEN, expand_x=True
    )

    # --- Área de seleção de arquivo (VISÍVEL) ---
    file_row = [
        sg.Text("Imagem de calibração:", size=(20, 1)),
        sg.Input(key="-FILE-", enable_events=True, expand_x=True, size=(1,1)),
        sg.FileBrowse("Escolher Imagem", target="-FILE-", file_types=(("Imagens", "*.png;*.jpg;*.jpeg"),), initial_folder="dataset"),
    ]

    # --- Coluna esquerda: pré-visualização ---
    left_col = [
        [sg.Image(key="-IMAGE-", size=PREVIEW_SIZE, expand_x=True, expand_y=True, background_color="#1b1b1b")]
    ]

    # --- Coluna direita: controles ---
    right_col = [
        [sg.Text("Poços detectados:")],
        [sg.Combo([], key="-WELL-SEL-", size=(28,1), enable_events=True, readonly=True),
        sg.Button("«", key="-PREV-"), sg.Button("»", key="-NEXT-")],
        [sg.Text("Concentração conhecida:"), sg.Input(key="-CONC-", size=(12, 1))],
        [sg.HorizontalSeparator()],
        [sg.Text("ROIs de Fundo (acumuladas):")],
        [sg.Listbox(values=[], key="-BG-LIST-", size=(42, 5), disabled=True, expand_x=True, expand_y=False)],
        [sg.Button("Adicionar ROI de Fundo (+)", key="-BG-ADD-", disabled=True),
        sg.Button("Remover Último Fundo", key="-BG-POP-", disabled=True)],
        [sg.Button("Limpar Fundos", key="-BG-CLEAR-", disabled=True)],
        [sg.HorizontalSeparator()],
        [adv_frame_calib],
        [sg.HorizontalSeparator()],
        [sg.Button("Adicionar Calibração", key="-ADD-", disabled=True),
        sg.Button("Calibrar Modelo", key="-CALIBRATE-", disabled=True),
        sg.Button("Exibir Curva", key="-PLOT-CURVE-", disabled=True)], 
        [sg.Button("Modo Análise", key="-SWITCH-ANALYSIS-"), sg.Button("Sair", key="-EXIT-")],
    ]

    # --- Layout principal (duas colunas) ---
    layout = [
        [sg.Text("Analisador de Imagens - Quantificação Bioquímica (Calibração)", font=("Any", 15))],
        file_row,
        [sg.Column(left_col, expand_x=True, expand_y=True),
         sg.Column(right_col, vertical_alignment="top", expand_x=True)],
        [sg.Text("Dados de calibração:")],
        [sg.Multiline(key="-DATA-", size=(120, 10), disabled=True, autoscroll=True, expand_x=True, expand_y=True)],
    ]

    window = sg.Window(
        "Quantificação Bioquímica - Calibração",
        layout,
        finalize=True,
        resizable=True,
        use_default_focus=False,
    )

    # Estado desta janela
    imagem_atual: Optional[Any] = None
    caminho_atual: Optional[str] = None
    circulos: List[Tuple[int,int,int]] = []
    scores: List[float] = []
    idx_sel: int = 0
    bg_rects: List[Tuple[int, int, int, int]] = []
    media_poco: Optional[float] = None
    media_fundo: Optional[float] = None
    media_corrigida: Optional[float] = None

    def _adv_vals(v: Dict[str, Any]) -> Dict[str, float | bool]:
        """Lê os valores do Painel Avançado (Calibração) a partir de 'values'."""
        return {
            "redet": bool(v["-ADV-REDET-"]),
            "photo": bool(v["-ADV-PHOTO-"]),
            "color": bool(v["-ADV-COLOR-"]),
            "param2": int(v["-ADV-PARAM2-"]),
            "abs_diff": float(v["-ADV-ABS-"]),
            "contrast": float(v["-ADV-CONTR-"]),
            "dE": float(v["-ADV-DE-"]),
            "smin": float(v["-ADV-SMIN-"]),
            "vdiff": float(v["-ADV-VDIFF-"]),
        }

    def _set_bg_buttons_state():
        has_img = imagem_atual is not None and bool(circulos)
        window["-BG-ADD-"].update(disabled=not has_img) #type: ignore
        window["-BG-POP-"].update(disabled=not bg_rects) #type: ignore
        window["-BG-CLEAR-"].update(disabled=not bg_rects) #type: ignore
        window["-BG-LIST-"].update(values=_fmt_bg_list(bg_rects)) #type: ignore

    def recalcular_medias():
        nonlocal media_poco, media_fundo, media_corrigida
        if imagem_atual is not None and circulos and 0 <= idx_sel < len(circulos):
            if bg_rects:
                m_p, m_f, m_c = calcular_medias_roi_multi_bg(imagem_atual, circulos[idx_sel], bg_rects)
            else:
                m_p, m_f, m_c = calcular_medias_roi(imagem_atual, circulos[idx_sel], bg_rect=None)
            media_poco, media_fundo, media_corrigida = m_p, m_f, m_c

    def atualizar_visualizacao():
        if imagem_atual is None or not circulos:
            return
        annotated = desenhar_overlays_multiplos(imagem_atual, circulos, idx_sel, bg_rects if bg_rects else None)
        window["-IMAGE-"].update(data=imagem_para_bytes(annotated, resize=PREVIEW_SIZE)) #type: ignore
        if media_poco is not None and media_fundo is not None and media_corrigida is not None:
            base = f"Imagem: {os.path.basename(caminho_atual) if caminho_atual else ''}\n"
            base += f"Poço selecionado: #{idx_sel+1}\n"
            base += f"Fundos selecionados: {len(bg_rects)}\n"
            base += f"Média poço: {media_poco:.2f}\nMédia fundo (média das ROIs): {media_fundo:.2f}\n"
            base += f"Valor corrigido (fundo - poço): {media_corrigida:.2f}\n"
            window["-DATA-"].update(base) #type: ignore

    def atualizar_combo():
        window["-WELL-SEL-"].update(values=_formatar_labels_pocos(circulos, scores)) #type: ignore
        labels = window["-WELL-SEL-"].Values #type: ignore
        if labels:
            window["-WELL-SEL-"].update(value=labels[idx_sel]) #type: ignore

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-EXIT-"):
            window.close()
            return None if event == sg.WIN_CLOSED else False

        elif event == "-FILE-":
            caminho = values["-FILE-"]
            if not caminho:
                continue
            try:
                imagem_atual = carregar_imagem(caminho)
                caminho_atual = caminho
                contexto_compartilhado["last_img_shape"] = imagem_atual.shape[:2]

                circulos = detectar_pocos(imagem_atual)
                if not circulos:
                    sg.popup("Nenhum poço detectado. Tente outra imagem ou ajuste a iluminação.", title="Aviso")
                    imagem_atual = None
                    window["-ADD-"].update(disabled=True) #type: ignore
                    window["-WELL-SEL-"].update(values=[], value="") #type: ignore
                    window["-IMAGE-"].update(data=None) #type: ignore
                    bg_rects.clear()
                    _set_bg_buttons_state()
                    continue

                scores = pontuar_pocos(imagem_atual, circulos)
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i]))
                bg_rects.clear()
                recalcular_medias()
                atualizar_combo()
                atualizar_visualizacao()

                window["-ADD-"].update(disabled=False) #type: ignore
                _set_bg_buttons_state()
            except Exception as e:
                sg.popup(f"Erro ao carregar imagem: {e}", title="Erro")

        elif event == "-PLOT-CURVE-":
                if not modelo.esta_calibrado():
                    sg.popup("Calibre o modelo primeiro para exibir a curva.", title="Aviso")
                    continue
                try:
                    modelo.plot_calibracao() 
                except Exception as e:
                    sg.popup(f"Erro ao exibir curva: {e}", title="Erro")

        elif event == "-WELL-SEL-":
            if imagem_atual is None or not circulos:
                continue
            label = values["-WELL-SEL-"]
            if label:
                idx_sel = _index_por_label(label)
                idx_sel = max(0, min(idx_sel, len(circulos)-1))
                recalcular_medias()
                atualizar_visualizacao()

        elif event == "-PREV-":
            if circulos:
                idx_sel = (idx_sel - 1) % len(circulos)
                atualizar_combo()
                recalcular_medias()
                atualizar_visualizacao()

        elif event == "-NEXT-":
            if circulos:
                idx_sel = (idx_sel + 1) % len(circulos)
                atualizar_combo()
                recalcular_medias()
                atualizar_visualizacao()


        elif event == "-BG-ADD-":
            if imagem_atual is None or not circulos:
                sg.popup("Carregue uma imagem e selecione/tenha poços detectados.", title="Aviso")
                continue
            rect = selecionar_roi_fundo_cv(imagem_atual)
            if rect is None:
                recalcular_medias()
                atualizar_visualizacao()
                continue

            bg_rects.append(rect)
            adv = _adv_vals(values)

            # Redetecção guiada e filtros conforme Painel Avançado
            if adv["redet"]:
                cs_novos = redetectar_pocos_guiado_por_fundos(
                    imagem_atual, bg_rects,
                    abs_diff_thresh=None,       # limiar adaptativo
                    param2=int(adv["param2"]),  # vindo do painel
                    ref_circles=circulos
                )
                if cs_novos:
                    circulos = cs_novos

            if adv["photo"]:
                filtrados = filtrar_pocos_por_fundos(
                    imagem_atual, circulos, bg_rects,
                    abs_diff_thresh=float(adv["abs_diff"]),
                    min_contrast_delta=float(adv["contrast"])
                )
                if filtrados:
                    circulos = filtrados

            if adv["color"] and bg_rects and circulos:
                circulos = filtrar_pocos_por_cor(
                    imagem_atual, circulos, bg_rects,
                    deltaE_min=float(adv["dE"]),
                    s_min=float(adv["smin"]),
                    v_diff_min=float(adv["vdiff"])
                )

            scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
            if circulos:
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i]))

            _set_bg_buttons_state()
            recalcular_medias()
            atualizar_combo()
            atualizar_visualizacao()

        elif event == "-BG-POP-":
            if not bg_rects:
                continue
            _ = bg_rects.pop()
            adv = _adv_vals(values)
            if imagem_atual is not None:
                if bg_rects:
                    if adv["redet"]:
                        cs_novos = redetectar_pocos_guiado_por_fundos(
                            imagem_atual, bg_rects, abs_diff_thresh=None,
                            param2=int(adv["param2"]), ref_circles=circulos
                        )
                        if cs_novos:
                            circulos = cs_novos
                    if adv["photo"]:
                        filtrados = filtrar_pocos_por_fundos(
                            imagem_atual, detectar_pocos(imagem_atual), bg_rects,
                            abs_diff_thresh=float(adv["abs_diff"]),
                            min_contrast_delta=float(adv["contrast"])
                        )
                        if filtrados:
                            circulos = filtrados
                    if adv["color"] and bg_rects and circulos:
                        circulos = filtrar_pocos_por_cor(
                            imagem_atual, circulos, bg_rects,
                            deltaE_min=float(adv["dE"]),
                            s_min=float(adv["smin"]),
                            v_diff_min=float(adv["vdiff"])
                        )
                else:
                    circulos = detectar_pocos(imagem_atual)

                scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i])) if scores else 0
            _set_bg_buttons_state()
            recalcular_medias()
            atualizar_combo()
            atualizar_visualizacao()

        elif event == "-BG-CLEAR-":
            bg_rects.clear()
            if imagem_atual is not None:
                circulos = detectar_pocos(imagem_atual)
                scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i])) if scores else 0
            _set_bg_buttons_state()
            recalcular_medias()
            atualizar_combo()
            atualizar_visualizacao()

        elif event == "-ADD-":
            if imagem_atual is None or media_corrigida is None:
                sg.popup("Não há dados de imagem. Carregue e processe uma imagem primeiro.", title="Aviso")
                continue
            conc_str = values["-CONC-"]
            try:
                if not conc_str:
                    raise ValueError("Informe a concentração conhecida.")
                conc_val = float(conc_str)
            except Exception as e:
                sg.popup(f"Concentração inválida: {e}", title="Erro")
                continue

            modelo.adicionar_dado(media_corrigida, conc_val)
            window["-DATA-"].update(
                f"Dado adicionado - (poço #{idx_sel+1}) Intensidade: {media_corrigida:.2f}, Concentração: {conc_val}\n",
                append=True
            )

            if modelo.total_dados() >= 2:
                window["-CALIBRATE-"].update(disabled=False)

        elif event == "-CALIBRATE-":
            try:
                if modelo.total_dados() < 2:
                    sg.popup("Adicione pelo menos 2 pontos de calibração.", title="Dados insuficientes")
                    continue
                modelo.calibrar()
                coefs, intercept = modelo.obter_coeficientes()
                r2 = modelo.obter_r2()
        
       
                if len(coefs) == 1:
                    msg = f"Modelo calibrado!\nConcentração = {coefs[0]:.4f} * Intensidade + {intercept:.4f}\nR² = {r2:.4f}"
                else:
            
                    coef_terms = " + ".join([f"{c:.4f}*X{i+1}" for i, c in enumerate(coefs)])
                    msg = f"Modelo calibrado!\nConcentração = {coef_terms} + {intercept:.4f}\nR² = {r2:.4f}"
        
                sg.popup(msg, title="Resultado da Calibração")
                window["-PLOT-CURVE-"].update(disabled=False) #type: ignore
            except Exception as e:
                sg.popup(f"Erro na calibração: {e}", title="Erro")

        elif event == "-SWITCH-ANALYSIS-":
            if not modelo.esta_calibrado():
                sg.popup("Calibre o modelo primeiro (mínimo de 2 pontos e 'Calibrar Modelo').", title="Aviso")
                continue
            # Persistir o que foi refinado na calibração
            contexto_compartilhado["bg_rects"] = bg_rects.copy()
            contexto_compartilhado["ref_circles"] = circulos.copy()
            contexto_compartilhado["last_img_shape"] = (imagem_atual.shape[:2] if imagem_atual is not None else None)
            window.close()
            return True

# ========================= ANÁLISE =========================

def run_analysis_window(modelo: ModeloCalibracao) -> Optional[bool]:
    sg.theme("LightBlue")

    # -------- Painel Avançado (Análise) --------
    adv_frame_anal = sg.Frame(
        "Avançado",
        [
            [sg.Checkbox("Redetectar guiado pelo fundo", key="-ADV-REDET-ANAL-", default=True),
             sg.Checkbox("Filtro fotométrico (fundo/contraste)", key="-ADV-PHOTO-ANAL-", default=True),
             sg.Checkbox("Filtro de cor (Lab/HSV)", key="-ADV-COLOR-ANAL-", default=True)],
            [sg.Text("param2 (Hough):", size=(16,1)),
             sg.Slider(range=(10,80), default_value=35, resolution=1, orientation="h", size=(30,15), key="-ADV-PARAM2-ANAL-")],
            [sg.Text("Δabs (disco−fundo):", size=(16,1)),
             sg.Slider(range=(0,25), default_value=6, resolution=1, orientation="h", size=(30,15), key="-ADV-ABS-ANAL-"),
             sg.Text("Δcontraste (anel−disco):", size=(22,1)),
             sg.Slider(range=(0,25), default_value=3, resolution=1, orientation="h", size=(30,15), key="-ADV-CONTR-ANAL-")],
            [sg.Text("ΔE_min (Lab):", size=(16,1)),
             sg.Slider(range=(0,25), default_value=8, resolution=1, orientation="h", size=(30,15), key="-ADV-DE-ANAL-"),
             sg.Text("S_min (HSV):", size=(12,1)),
             sg.Slider(range=(0,255), default_value=15, resolution=1, orientation="h", size=(30,15), key="-ADV-SMIN-ANAL-"),
             sg.Text("|V−Vfundo|:", size=(10,1)),
             sg.Slider(range=(0,50), default_value=8, resolution=1, orientation="h", size=(30,15), key="-ADV-VDIFF-ANAL-")],
        ],
        relief=sg.RELIEF_SUNKEN, expand_x=True
    )

    # --- Área de seleção de arquivo (VISÍVEL) ---
    file_row = [
        sg.Text("Imagem para análise:", size=(20, 1)),
        sg.Input(key="-FILE-ANAL-", enable_events=True, expand_x=True, size=(1,1)),
        sg.FileBrowse("Escolher Imagem", target="-FILE-ANAL-", file_types=(("Imagens", "*.png;*.jpg;*.jpeg"),), initial_folder="dataset"),
    ]

    # --- Coluna esquerda: pré-visualização ---
    left_col = [
        [sg.Image(key="-IMAGE-ANAL-", size=PREVIEW_SIZE, expand_x=True, expand_y=True, background_color="#1b1b1b")]
    ]

    # --- Coluna direita: controles ---
    right_col = [
        [sg.Text("Poços detectados:")],
        [sg.Combo([], key="-WELL-SEL-ANAL-", size=(28,1), enable_events=True, readonly=True),
         sg.Button("«", key="-PREV-ANAL-"), sg.Button("»", key="-NEXT-ANAL-")],
        [sg.Text("Concentração prevista:"), sg.Input(key="-PRED-", size=(15, 1), disabled=True)],
        [sg.HorizontalSeparator()],
        [sg.Text("ROIs de Fundo (acumuladas):")],
        [sg.Listbox(values=[], key="-BG-LIST-ANAL-", size=(42, 5), disabled=True, expand_x=True, expand_y=False)],
        [sg.Button("Adicionar ROI de Fundo (+)", key="-BG-ADD-ANAL-", disabled=True),
         sg.Button("Remover Último Fundo", key="-BG-POP-ANAL-", disabled=True)],
        [sg.Button("Limpar Fundos", key="-BG-CLEAR-ANAL-", disabled=True)],
        [sg.HorizontalSeparator()],
        [adv_frame_anal],
        [sg.HorizontalSeparator()],
        [sg.Text("Processar pasta (lote):")],
        [sg.Input(key="-FOLDER-", enable_events=False, expand_x=True), sg.FolderBrowse("Selecionar Pasta", initial_folder="dataset"),
         sg.Button("Processar Pasta", key="-BATCH-")],
        [sg.Button("Voltar para Calibração", key="-BACK-"), sg.Button("Sair", key="-EXIT-")],
    ]

    layout = [
        [sg.Text("Analisador de Imagens - Quantificação Bioquímica (Análise)", font=("Any", 15))],
        file_row,
        [sg.Column(left_col, expand_x=True, expand_y=True),
         sg.Column(right_col, vertical_alignment="top", expand_x=True)],
        [sg.Multiline(key="-RES-", size=(120, 12), disabled=True, autoscroll=True, expand_x=True, expand_y=True)],
    ]

    window = sg.Window(
        "Quantificação Bioquímica - Análise",
        layout,
        finalize=True,
        resizable=True,
        use_default_focus=False,
    )

    # Estado desta janela
    imagem_atual: Optional[Any] = None
    caminho_atual: Optional[str] = None
    circulos: List[Tuple[int,int,int]] = []
    scores: List[float] = []
    idx_sel: int = 0
    bg_rects: List[Tuple[int, int, int, int]] = []
    media_poco: Optional[float] = None
    media_fundo: Optional[float] = None
    media_corrigida: Optional[float] = None

    def _adv_vals(v: Dict[str, Any]) -> Dict[str, float | bool]:
        """Lê os valores do Painel Avançado (Análise) a partir de 'values'."""
        return {
            "redet": bool(v["-ADV-REDET-ANAL-"]),
            "photo": bool(v["-ADV-PHOTO-ANAL-"]),
            "color": bool(v["-ADV-COLOR-ANAL-"]),
            "param2": int(v["-ADV-PARAM2-ANAL-"]),
            "abs_diff": float(v["-ADV-ABS-ANAL-"]),
            "contrast": float(v["-ADV-CONTR-ANAL-"]),
            "dE": float(v["-ADV-DE-ANAL-"]),
            "smin": float(v["-ADV-SMIN-ANAL-"]),
            "vdiff": float(v["-ADV-VDIFF-ANAL-"]),
        }

    def _set_bg_buttons_state():
        has_img = imagem_atual is not None and bool(circulos)
        window["-BG-ADD-ANAL-"].update(disabled=not has_img) #type: ignore
        window["-BG-POP-ANAL-"].update(disabled=not bg_rects) #type: ignore
        window["-BG-CLEAR-ANAL-"].update(disabled=not bg_rects) #type: ignore
        window["-BG-LIST-ANAL-"].update(values=_fmt_bg_list(bg_rects)) #type: ignore

    def recalcular_medias():
        nonlocal media_poco, media_fundo, media_corrigida
        if imagem_atual is not None and circulos and 0 <= idx_sel < len(circulos):
            if bg_rects:
                m_p, m_f, m_c = calcular_medias_roi_multi_bg(imagem_atual, circulos[idx_sel], bg_rects)
            else:
                m_p, m_f, m_c = calcular_medias_roi(imagem_atual, circulos[idx_sel], bg_rect=None)
            media_poco, media_fundo, media_corrigida = m_p, m_f, m_c

    def atualizar_visualizacao():
        if imagem_atual is None or not circulos:
            return
        annotated = desenhar_overlays_multiplos(imagem_atual, circulos, idx_sel, bg_rects if bg_rects else None)
        window["-IMAGE-ANAL-"].update(data=imagem_para_bytes(annotated, resize=PREVIEW_SIZE)) #type: ignore

    def atualizar_combo():
        window["-WELL-SEL-ANAL-"].update(values=_formatar_labels_pocos(circulos, scores)) #type: ignore
        labels = window["-WELL-SEL-ANAL-"].Values #type: ignore
        if labels:
            window["-WELL-SEL-ANAL-"].update(value=labels[idx_sel]) #type: ignore

    def prever_e_exibir():
        if media_corrigida is None:
            return
        try:
            conc_pred = modelo.prever(media_corrigida)
            window["-PRED-"].update(f"{conc_pred:.2f}") #type: ignore
            base = ""
            if caminho_atual:
                base += f"Imagem: {os.path.basename(caminho_atual)}\n"
            base += (f"(poço #{idx_sel+1}) Fundos: {len(bg_rects)} | "
                     f"Média poço: {media_poco:.2f} | "
                     f"Média fundo: {media_fundo:.2f} | "
                     f"Valor corrigido: {media_corrigida:.2f} | "
                     f"Concentração prevista: {conc_pred:.2f}\n")
            window["-RES-"].update(base, append=True) #type: ignore
        except Exception as e:
            sg.popup(f"Erro na predição: {e}", title="Erro")

    def _escalar_rects(rects, src_shape, dst_shape):
        """Escala uma lista de retângulos (x,y,w,h) de src_shape para dst_shape."""
        if not rects or src_shape is None or dst_shape is None:
            return rects
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape
        if src_h == 0 or src_w == 0:
            return rects
        sy = dst_h / float(src_h)
        sx = dst_w / float(src_w)
        scaled = []
        for (x, y, w, h) in rects:
            nx = int(round(x * sx))
            ny = int(round(y * sy))
            nw = int(round(w * sx))
            nh = int(round(h * sy))
            scaled.append((nx, ny, max(1, nw), max(1, nh)))
        return scaled

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-EXIT-"):
            window.close()
            return None if event == sg.WIN_CLOSED else False

        elif event == "-BACK-":
            window.close()
            return True

        elif event == "-FILE-ANAL-":
            caminho = values["-FILE-ANAL-"]
            if not caminho:
                continue
            try:
                imagem_atual = carregar_imagem(caminho)
                caminho_atual = caminho

                circulos = detectar_pocos(imagem_atual)
                if not circulos:
                    sg.popup("Nenhum poço detectado nesta imagem.", title="Aviso")
                    imagem_atual = None
                    window["-PRED-"].update("") #type: ignore
                    bg_rects.clear()
                    _set_bg_buttons_state()
                    continue

                scores = pontuar_pocos(imagem_atual, circulos)
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i]))
                bg_rects.clear()

                # ----- APLICAR CONTEXTO DA CALIBRAÇÃO -----
                adv = _adv_vals(values)
                bg_rects_calib = contexto_compartilhado.get("bg_rects", [])
                ref_circles_calib = contexto_compartilhado.get("ref_circles", [])
                shape_calib = contexto_compartilhado.get("last_img_shape", None)
                if bg_rects_calib:
                    bg_rects = _escalar_rects(bg_rects_calib, shape_calib, imagem_atual.shape[:2])
                    window["-BG-LIST-ANAL-"].update(values=_fmt_bg_list(bg_rects)) #type: ignore

                    if adv["redet"]:
                        cs_novos = redetectar_pocos_guiado_por_fundos(
                            imagem_atual, bg_rects,
                            abs_diff_thresh=None,
                            param2=int(adv["param2"]),
                            ref_circles=(ref_circles_calib if ref_circles_calib else circulos)
                        )
                        if cs_novos:
                            circulos = cs_novos
                    if adv["photo"]:
                        filtrados = filtrar_pocos_por_fundos(
                            imagem_atual, circulos, bg_rects,
                            abs_diff_thresh=float(adv["abs_diff"]),
                            min_contrast_delta=float(adv["contrast"])
                        )
                        if filtrados:
                            circulos = filtrados
                    if adv["color"] and circulos:
                        circulos = filtrar_pocos_por_cor(
                            imagem_atual, circulos, bg_rects,
                            deltaE_min=float(adv["dE"]),
                            s_min=float(adv["smin"]),
                            v_diff_min=float(adv["vdiff"])
                        )

                scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
                if circulos:
                    idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i]))

                _set_bg_buttons_state()
                recalcular_medias()
                atualizar_combo()
                atualizar_visualizacao()
                prever_e_exibir()
            except Exception as e:
                sg.popup(f"Erro ao carregar/analisar: {e}", title="Erro")

        elif event == "-WELL-SEL-ANAL-":
            if imagem_atual is None or not circulos:
                continue
            label = values["-WELL-SEL-ANAL-"]
            if label:
                idx_sel = _index_por_label(label)
                idx_sel = max(0, min(idx_sel, len(circulos)-1))
                recalcular_medias()
                atualizar_visualizacao()
                prever_e_exibir()

        elif event == "-PREV-ANAL-":
            if circulos:
                idx_sel = (idx_sel - 1) % len(circulos)
                atualizar_combo()
                recalcular_medias()
                atualizar_visualizacao()
                prever_e_exibir()

        elif event == "-NEXT-ANAL-":
            if circulos:
                idx_sel = (idx_sel + 1) % len(circulos)
                atualizar_combo()
                recalcular_medias()
                atualizar_visualizacao()
                prever_e_exibir()


        elif event == "-BG-ADD-ANAL-":
            if imagem_atual is None or not circulos:
                sg.popup("Carregue uma imagem e selecione/tenha poços detectados.", title="Aviso")
                continue
            rect = selecionar_roi_fundo_cv(imagem_atual)
            if rect is None:
                recalcular_medias()
                atualizar_visualizacao()
                prever_e_exibir()
                continue

            bg_rects.append(rect)
            adv = _adv_vals(values)

            if adv["redet"]:
                cs_novos = redetectar_pocos_guiado_por_fundos(
                    imagem_atual, bg_rects,
                    abs_diff_thresh=None,
                    param2=int(adv["param2"]),
                    ref_circles=circulos
                )
                if cs_novos:
                    circulos = cs_novos
            if adv["photo"]:
                filtrados = filtrar_pocos_por_fundos(
                    imagem_atual, circulos, bg_rects,
                    abs_diff_thresh=float(adv["abs_diff"]),
                    min_contrast_delta=float(adv["contrast"])
                )
                if filtrados:
                    circulos = filtrados
            if adv["color"] and bg_rects and circulos:
                circulos = filtrar_pocos_por_cor(
                    imagem_atual, circulos, bg_rects,
                    deltaE_min=float(adv["dE"]),
                    s_min=float(adv["smin"]),
                    v_diff_min=float(adv["vdiff"])
                )

            scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
            if circulos:
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i]))

            _set_bg_buttons_state()
            recalcular_medias()
            atualizar_combo()
            atualizar_visualizacao()
            prever_e_exibir()

        elif event == "-BG-POP-ANAL-":
            if not bg_rects:
                continue
            _ = bg_rects.pop()
            adv = _adv_vals(values)
            if imagem_atual is not None:
                if bg_rects:
                    if adv["redet"]:
                        cs_novos = redetectar_pocos_guiado_por_fundos(
                            imagem_atual, bg_rects, abs_diff_thresh=None,
                            param2=int(adv["param2"]), ref_circles=circulos
                        )
                        if cs_novos:
                            circulos = cs_novos
                    if adv["photo"]:
                        filtrados = filtrar_pocos_por_fundos(
                            imagem_atual, detectar_pocos(imagem_atual), bg_rects,
                            abs_diff_thresh=float(adv["abs_diff"]),
                            min_contrast_delta=float(adv["contrast"])
                        )
                        if filtrados:
                            circulos = filtrados
                    if adv["color"] and bg_rects and circulos:
                        circulos = filtrar_pocos_por_cor(
                            imagem_atual, circulos, bg_rects,
                            deltaE_min=float(adv["dE"]),
                            s_min=float(adv["smin"]),
                            v_diff_min=float(adv["vdiff"])
                        )
                else:
                    circulos = detectar_pocos(imagem_atual)
                scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i])) if scores else 0
            _set_bg_buttons_state()
            recalcular_medias()
            atualizar_combo()
            atualizar_visualizacao()
            prever_e_exibir()

        elif event == "-BG-CLEAR-ANAL-":
            bg_rects.clear()
            if imagem_atual is not None:
                circulos = detectar_pocos(imagem_atual)
                scores = pontuar_pocos(imagem_atual, circulos) if circulos else []
                idx_sel = int(max(range(len(circulos)), key=lambda i: scores[i])) if scores else 0
            _set_bg_buttons_state()
            recalcular_medias()
            atualizar_combo()
            atualizar_visualizacao()
            prever_e_exibir()

        elif event == "-BATCH-":
            pasta = values.get("-FOLDER-")
            if not pasta or not os.path.isdir(pasta):
                sg.popup("Selecione uma pasta válida.", title="Aviso")
                continue
            if not modelo.esta_calibrado():
                sg.popup("Calibre o modelo antes de processar em lote.", title="Aviso")
                continue

            resultados = []
            for nome in sorted(os.listdir(pasta)):
                if not nome.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                caminho_img = os.path.join(pasta, nome)
                try:
                    img = carregar_imagem(caminho_img)
                    cs = detectar_pocos(img)
                    if not cs:
                        resultados.append((nome, None))
                        window["-RES-"].update(f"{nome}: poços não detectados\n", append=True) #type: ignore
                        continue
                    scs = pontuar_pocos(img, cs)
                    j = int(max(range(len(cs)), key=lambda i: scs[i]))
                    _, _, m_c = calcular_medias_roi(img, cs[j], bg_rect=None)
                    conc = modelo.prever(m_c)
                    resultados.append((nome, conc))
                    window["-RES-"].update(f"{nome}: {conc:.2f}\n", append=True) #type: ignore
                except Exception as e:
                    resultados.append((nome, None))
                    window["-RES-"].update(f"{nome}: erro ({e})\n", append=True) #type: ignore

            csv_path = os.path.join(pasta, "resultados_analise.csv")
            try:
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("Imagem,Concentracao_Prevista\n")
                    for nome, conc in resultados:
                        f.write(f"{nome},{'N/A' if conc is None else f'{conc:.6f}'}\n")
                sg.popup(f"Processamento concluído.\nCSV salvo em:\n{csv_path}", title="Concluído")
            except Exception as e:
                sg.popup(f"Falha ao salvar CSV: {e}", title="Erro")
