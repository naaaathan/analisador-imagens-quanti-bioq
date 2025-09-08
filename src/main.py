#!/usr/bin/env python3
# main.py: Ponto de entrada do programa (Requisito 8 - estrutura modular).
# Este arquivo inicia a aplicação de interface gráfica (Requisito 1).
"""
Arquivo principal que inicializa a aplicação de análise de imagens bioquímicas.
Requisitos implementados neste módulo:
 - Requisito 1: Utilizar interface gráfica com PySimpleGUI (inicialização da GUI).
 - Requisito 8: Estrutura do projeto modularizada em vários arquivos (este é o ponto de entrada).
"""
from gui import iniciar_interface  # Importa a função que inicia a interface gráfica (GUI)

if __name__ == "__main__":
    # Executa a interface gráfica do aplicativo.
    iniciar_interface()
