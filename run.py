#!/usr/bin/env python3
"""
Script para executar o Sistema de Exploração Multiagente Colaborativo
"""

import subprocess
import sys
import os

def main():
    # Mudar para o diretório do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Executar Streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "Core/app.py", "--server.headless", "true"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar Streamlit: {e}")
    except KeyboardInterrupt:
        print("Aplicação parada pelo utilizador")

if __name__ == "__main__":
    main()