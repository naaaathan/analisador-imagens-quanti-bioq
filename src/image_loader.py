#!/usr/bin/env python3
# image_loader.py — Responsabilidade: carregar imagens a partir do disco.

import cv2

def carregar_imagem(caminho: str):
    """Lê a imagem do disco em BGR (OpenCV)."""
    img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho}")
    return img
