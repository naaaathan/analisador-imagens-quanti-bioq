#!/usr/bin/env python3
# gui_extended.py — Pequena camada encima de gui.py para adicionar seletores de Modelo de Cor, Componentes, Compressão e Nanômetros.
import os
import PySimpleGUI as sg
from gui import iniciar_interface, modelo_global, contexto_compartilhado
from utils import save_compressed_versions, extract_features, spectral_curve_from_roi
from regression_model import ModeloCalibracao

def settings_window():
    sg.theme("LightBlue")
    layout = [
        [sg.Text("Configurações do Projeto")],
        [sg.Text("Modelo de cor:"), sg.Combo(["RGB","HSV","LAB"], default_value="RGB", key="-COLOR-")],
        [sg.Text("Componentes (para calibração/análise):"),
         sg.Combo(["ALL","H","S","V","R","G","B","L","a","b","b*"], default_value="ALL", key="-COMP-")],
        [sg.Text("Pasta para compressão (opcional):"), sg.Input(key="-SRC-"), sg.FolderBrowse()],
        [sg.Text("Destino compressão:"), sg.Input(key="-DST-"), sg.FolderBrowse()],
        [sg.Button("Salvar e Abrir App"), sg.Button("Cancelar")]
    ]
    win = sg.Window("Configurações", layout, finalize=True)
    event, vals = win.read()
    if event == "Salvar e Abrir App":
        color = vals["-COLOR-"]
        comp = vals["-COMP-"]
        src = vals["-SRC-"]
        dst = vals["-DST-"]
        win.close()
        # executar compressão se solicitado
        if src and dst:
            try:
                save_compressed_versions(src, dst)
                sg.popup("Compressão concluída.", title="Compressão")
            except Exception as e:
                sg.popup(f"Compressão falhou: {e}", title="Erro")
        # armazenar escolhas no contexto compartilhado
        contexto_compartilhado["color_model"] = color
        contexto_compartilhado["components"] = ([comp] if comp!="ALL" else ["ALL"])
        # iniciar interface principal
        iniciar_interface()
    else:
        win.close()

if __name__ == '__main__':
    settings_window()
