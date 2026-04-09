"""
Script para organizar o dataset em pastas por classe.

Estrutura esperada:
    dataset/
    +-- folhagem/      <- Imagens da folhagem do abacaxi (com ou sem fruto)
    +-- fruto_sozinho/  <- Imagens do abacaxi/fruto sozinho (sem folhagem)

Uso:
    python organize_dataset.py
"""

import os
import sys
import shutil

# Forcar UTF-8 no stdout (Windows)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def organizar_dataset():
    """
    Organiza as imagens da pasta 'images/' na estrutura de pastas do dataset.
    Todas as imagens existentes sao classificadas como 'folhagem' (planta com folhas).
    
    IMPORTANTE: Voce deve adicionar manualmente imagens de abacaxi/fruto sozinho
    (sem a planta/folhagem) na pasta 'dataset/fruto_sozinho/'.
    """
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGES_DIR = os.path.join(BASE_DIR, "images")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    
    # Classes do modelo
    CLASSE_FOLHAGEM = os.path.join(DATASET_DIR, "folhagem")
    CLASSE_FRUTO = os.path.join(DATASET_DIR, "fruto_sozinho")
    
    # Criar estrutura de pastas
    os.makedirs(CLASSE_FOLHAGEM, exist_ok=True)
    os.makedirs(CLASSE_FRUTO, exist_ok=True)
    
    # Copiar imagens de folhagem
    if not os.path.exists(IMAGES_DIR):
        print(f"[ERRO] Pasta '{IMAGES_DIR}' nao encontrada!")
        return
    
    imagens = [f for f in os.listdir(IMAGES_DIR) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    if not imagens:
        print("[ERRO] Nenhuma imagem encontrada na pasta 'images/'!")
        return
    
    copiadas = 0
    for img in imagens:
        origem = os.path.join(IMAGES_DIR, img)
        destino = os.path.join(CLASSE_FOLHAGEM, img)
        if not os.path.exists(destino):
            shutil.copy2(origem, destino)
            copiadas += 1
            print(f"  [OK] {img} -> dataset/folhagem/")
        else:
            print(f"  [SKIP] {img} ja existe em dataset/folhagem/")
    
    # Contar imagens em cada classe
    n_folhagem = len([f for f in os.listdir(CLASSE_FOLHAGEM) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
    n_fruto = len([f for f in os.listdir(CLASSE_FRUTO) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
    
    print("\n" + "=" * 60)
    print("RESUMO DO DATASET")
    print("=" * 60)
    print(f"  dataset/folhagem/      -> {n_folhagem} imagens")
    print(f"  dataset/fruto_sozinho/ -> {n_fruto} imagens")
    print("=" * 60)
    
    if n_fruto == 0:
        print("\n[ATENCAO] A pasta 'dataset/fruto_sozinho/' esta vazia!")
        print("   Para treinar o modelo, voce precisa adicionar imagens de")
        print("   abacaxi/fruto SOZINHO (sem a folhagem da planta).")
        print("   Adicione pelo menos 5-10 imagens para melhores resultados.")
    
    if n_folhagem > 0 and n_fruto > 0:
        print("\n[OK] Dataset pronto! Execute 'python train_model.py' para treinar.")


if __name__ == "__main__":
    print("Organizador de Dataset - Abacaxi")
    print("-" * 40)
    organizar_dataset()
