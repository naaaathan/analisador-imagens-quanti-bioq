#!/usr/bin/env python3
# regression_model.py — Responsabilidade: gerenciar dados de calibração e regressão linear.

from typing import List, Tuple
import numpy as np

class ModeloCalibracao:
    """
    Requisito (Calibração):
      - Armazenar pares (intensidade_corrigida, concentração_conhecida).
      - Ajustar um modelo linear: concentração = a * intensidade + b
      - Prever concentração para novas intensidades.
    """
    def __init__(self):
        self._xs: List[float] = []  # intensidades corrigidas
        self._ys: List[float] = []  # concentrações conhecidas
        self._a: float = 0.0
        self._b: float = 0.0
        self._r2: float = 0.0
        self._calibrado: bool = False

    def adicionar_dado(self, intensidade_corrigida: float, concentracao: float) -> None:
        self._xs.append(float(intensidade_corrigida))
        self._ys.append(float(concentracao))
        self._calibrado = False  # precisa recalibrar

    def total_dados(self) -> int:
        return len(self._xs)

    def calibrar(self) -> None:
        if len(self._xs) < 2:
            raise ValueError("É necessário pelo menos 2 pontos de calibração.")
        x = np.array(self._xs, dtype=float)
        y = np.array(self._ys, dtype=float)
        # Ajuste linear y = a*x + b
        a, b = np.polyfit(x, y, 1)
        y_pred = a * x + b
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        self._a, self._b, self._r2 = float(a), float(b), float(r2)
        self._calibrado = True

    def prever(self, intensidade_corrigida: float) -> float:
        if not self._calibrado:
            raise RuntimeError("Modelo não calibrado. Adicione dados e clique em 'Calibrar Modelo'.")
        return self._a * float(intensidade_corrigida) + self._b

    def obter_coeficientes(self) -> Tuple[float, float]:
        if not self._calibrado:
            raise RuntimeError("Modelo não calibrado.")
        return self._a, self._b

    def obter_r2(self) -> float:
        if not self._calibrado:
            raise RuntimeError("Modelo não calibrado.")
        return self._r2

    def esta_calibrado(self) -> bool:
        return self._calibrado
