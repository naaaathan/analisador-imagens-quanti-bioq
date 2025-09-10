#!/usr/bin/env python3
# regression_model.py – Responsabilidade: gerenciar dados de calibração e regressão linear multi-feature.
from typing import List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

class ModeloCalibracao:
    """
    Suporta X multivariável (p.ex. H,S,V ou L,a,b) e regressão linear multivariada.
    - adicionar_dado(x_vector, y_scalar) onde x_vector pode ser a intensidade ou um vetor de features
    - calibrar() ajusta LinearRegression
    - prever(x_vector) retorna escalar previsão
    - plot_calibracao(path=None) plota y_true vs y_pred e salva/mostra
    """
    def __init__(self):
        self._Xs: List[List[float]] = []  # lista de vetores de features
        self._ys: List[float] = []        # concentrações conhecidas
        self.model: Optional[LinearRegression] = None
        self._r2: float = 0.0
        self._calibrado: bool = False

    def adicionar_dado(self, intensidade_corrigida, concentracao: float) -> None:
        # aceita escalar ou iterável de features
        if hasattr(intensidade_corrigida, "__iter__") and not isinstance(intensidade_corrigida, (str, bytes)):
            vec = [float(x) for x in intensidade_corrigida]
        else:
            vec = [float(intensidade_corrigida)]
        self._Xs.append(vec)
        self._ys.append(float(concentracao))
        self._calibrado = False

    def total_dados(self) -> int:
        return len(self._ys)

    def calibrar(self) -> None:
        if len(self._ys) < 2:
            raise ValueError("É necessário pelo menos 2 pontos de calibração.")
        X = np.array(self._Xs, dtype=float)
        y = np.array(self._ys, dtype=float)
        self.model = LinearRegression()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        self._r2 = float(r2)
        self._calibrado = True

    def prever(self, intensidade_corrigida) -> float:
        if not self._calibrado or self.model is None:
            raise RuntimeError("Modelo não calibrado. Adicione dados e clique em 'Calibrar Modelo'.")
        if hasattr(intensidade_corrigida, "__iter__") and not isinstance(intensidade_corrigida, (str, bytes)):
            x = np.array([float(x) for x in intensidade_corrigida], dtype=float).reshape(1, -1)
        else:
            x = np.array([float(intensidade_corrigida)], dtype=float).reshape(1, -1)
        return float(self.model.predict(x)[0])

    def obter_coeficientes(self) -> Tuple[List[float], float]:
        """
        Retorna os coeficientes como lista de floats e o intercepto como float.
        Para regressão multivariável: y = a1*x1 + a2*x2 + ... + b
        """
        if not self._calibrado or self.model is None:
            raise RuntimeError("Modelo não calibrado.")
        # Retorna coeficientes e intercepto: y = sum(a_i * x_i) + b
        coefs = [float(c) for c in self.model.coef_]
        intercept = float(self.model.intercept_)
        return coefs, intercept

    def obter_r2(self) -> float:
        if not self._calibrado:
            raise RuntimeError("Modelo não calibrado.")
        return self._r2

    def esta_calibrado(self) -> bool:
        return self._calibrado

    def plot_calibracao(self, path: Optional[str] = None) -> None:
        if not self._calibrado or self.model is None:
            raise RuntimeError("Modelo não calibrado.")
        X = np.array(self._Xs, dtype=float)
        y = np.array(self._ys, dtype=float)
        y_pred = self.model.predict(X)
        plt.figure(figsize=(6,6))
        plt.scatter(y, y_pred, alpha=0.7)
        mn = min(min(y), min(y_pred))
        mx = max(max(y), max(y_pred))
        plt.plot([mn, mx], [mn, mx], linestyle='--')
        plt.xlabel("Concentração observada")
        plt.ylabel("Concentração prevista")
        plt.title(f"Calibração (R² = {self._r2:.4f})")
        plt.grid(True)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()