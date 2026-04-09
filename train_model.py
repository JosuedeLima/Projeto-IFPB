"""
Script de Treinamento - Classificador de Folhagem de Abacaxi

Modelo de Machine Learning para identificar a folhagem do abacaxi (com ou sem fruto)
e diferenciar de imagens que contem apenas o fruto sozinho (sem folhagem).

Classificacao binaria:
    - Classe 0: folhagem      -> Planta do abacaxi com folhas (com ou sem fruto visivel)
    - Classe 1: fruto_sozinho -> Abacaxi/fruto isolado (sem a planta/folhagem)

Estrategia:
    - Transfer Learning com MobileNetV2 (pre-treinado no ImageNet)
    - Data Augmentation agressivo para compensar dataset pequeno
    - Fine-tuning das camadas finais

Uso:
    python train_model.py

Pre-requisitos:
    pip install tensorflow Pillow numpy matplotlib scikit-learn
"""

import os
import sys
import numpy as np

# Forcar UTF-8 no stdout (Windows)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # Backend nao-interativo para salvar graficos
import matplotlib.pyplot as plt

from PIL import Image

# ============================================================================
# CONFIGURACOES
# ============================================================================

# Diretorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "modelo")

# Parametros do modelo
IMG_SIZE = 224          # MobileNetV2 espera 224x224
BATCH_SIZE = 8          # Batch pequeno para dataset pequeno
EPOCHS = 50             # Epocas de treinamento (early stopping interrompe antes)
LEARNING_RATE = 0.0001  # Taxa de aprendizado baixa para fine-tuning
AUGMENTATION_FACTOR = 30  # Quantas variacoes gerar por imagem original

# Classes
CLASSES = ["folhagem", "fruto_sozinho"]


# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def verificar_dataset():
    """Verifica se o dataset esta organizado corretamente."""
    print("\n[INFO] Verificando dataset...")
    
    if not os.path.exists(DATASET_DIR):
        print("[ERRO] Pasta 'dataset/' nao encontrada!")
        print("   Execute primeiro: python organize_dataset.py")
        sys.exit(1)
    
    contagem = {}
    for classe in CLASSES:
        pasta = os.path.join(DATASET_DIR, classe)
        if not os.path.exists(pasta):
            print(f"[ERRO] Pasta 'dataset/{classe}/' nao encontrada!")
            print("   Execute primeiro: python organize_dataset.py")
            sys.exit(1)
        
        imgs = [f for f in os.listdir(pasta) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        contagem[classe] = len(imgs)
        print(f"   {classe}: {len(imgs)} imagens")
    
    for classe, n in contagem.items():
        if n == 0:
            print(f"\n[ERRO] A classe '{classe}' nao tem imagens!")
            if classe == "fruto_sozinho":
                print("   Adicione imagens de abacaxi sozinho (sem folhagem)")
                print("   na pasta 'dataset/fruto_sozinho/'")
            sys.exit(1)
    
    if min(contagem.values()) < 2:
        print("\n[AVISO] Pelo menos uma classe tem menos de 2 imagens.")
        print("   O modelo pode nao generalizar bem. Adicione mais imagens.")
    
    return contagem


def carregar_imagens():
    """Carrega todas as imagens do dataset e retorna arrays numpy."""
    print("\n[INFO] Carregando imagens...")
    
    imagens = []
    labels = []
    nomes = []
    
    for idx, classe in enumerate(CLASSES):
        pasta = os.path.join(DATASET_DIR, classe)
        arquivos = [f for f in os.listdir(pasta) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        for arquivo in arquivos:
            caminho = os.path.join(pasta, arquivo)
            try:
                img = Image.open(caminho).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                img_array = np.array(img, dtype=np.float32)
                
                imagens.append(img_array)
                labels.append(idx)
                nomes.append(f"{classe}/{arquivo}")
                
            except Exception as e:
                print(f"   [AVISO] Erro ao carregar {arquivo}: {e}")
    
    X = np.array(imagens)
    y = np.array(labels)
    
    print(f"   [OK] {len(X)} imagens carregadas ({IMG_SIZE}x{IMG_SIZE})")
    
    return X, y, nomes


def criar_augmentacoes(X, y, fator=AUGMENTATION_FACTOR):
    """
    Gera imagens aumentadas usando transformacoes aleatorias.
    Como as imagens sao vistas aereas, rotacoes e flips sao perfeitamente validos.
    """
    print(f"\n[INFO] Gerando data augmentation (x{fator} por imagem)...")
    
    X_aug = list(X.copy())
    y_aug = list(y.copy())
    
    for i in range(len(X)):
        img = X[i]
        label = y[i]
        
        for _ in range(fator):
            img_aug = img.copy()
            
            # Rotacao aleatoria (0-360 graus - valido para vistas aereas)
            k = np.random.randint(0, 4)
            img_aug = np.rot90(img_aug, k)
            
            # Flip horizontal aleatorio
            if np.random.random() > 0.5:
                img_aug = np.fliplr(img_aug)
            
            # Flip vertical aleatorio
            if np.random.random() > 0.5:
                img_aug = np.flipud(img_aug)
            
            # Ajuste de brilho aleatorio
            brilho = np.random.uniform(0.7, 1.3)
            img_aug = np.clip(img_aug * brilho, 0, 255)
            
            # Ajuste de contraste aleatorio
            contraste = np.random.uniform(0.8, 1.2)
            media = np.mean(img_aug)
            img_aug = np.clip((img_aug - media) * contraste + media, 0, 255)
            
            # Ruido gaussiano leve
            if np.random.random() > 0.5:
                ruido = np.random.normal(0, 5, img_aug.shape)
                img_aug = np.clip(img_aug + ruido, 0, 255)
            
            # Ajuste de saturacao (via conversao simples)
            if np.random.random() > 0.5:
                gray = np.mean(img_aug, axis=2, keepdims=True)
                sat = np.random.uniform(0.7, 1.3)
                img_aug = np.clip(gray + (img_aug - gray) * sat, 0, 255)
            
            X_aug.append(img_aug.astype(np.float32))
            y_aug.append(label)
    
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    print(f"   [OK] Dataset expandido: {len(X)} -> {len(X_aug)} imagens")
    
    # Contar por classe
    for idx, classe in enumerate(CLASSES):
        n = np.sum(y_aug == idx)
        print(f"      {classe}: {n} imagens")
    
    return X_aug, y_aug


def construir_modelo():
    """
    Constroi o modelo usando Transfer Learning com MobileNetV2.
    
    MobileNetV2 e leve, rapido e funciona bem com datasets pequenos
    gracas as features pre-aprendidas no ImageNet.
    """
    print("\n[INFO] Construindo modelo (MobileNetV2 + Transfer Learning)...")
    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    
    # Carregar MobileNetV2 pre-treinado (sem as camadas de classificacao)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congelar as camadas do modelo base
    base_model.trainable = False
    
    # Construir modelo completo
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Pre-processamento (normalizacao do MobileNetV2)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Modelo base (feature extractor)
    x = base_model(x, training=False)
    
    # Camadas de classificacao customizadas
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Classificacao binaria
    
    modelo = models.Model(inputs, outputs)
    
    # Compilar
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    modelo.summary()
    
    return modelo, base_model


def fine_tune_modelo(modelo, base_model):
    """
    Descongela as ultimas camadas do modelo base para fine-tuning.
    Isso permite que o modelo ajuste features de alto nivel para nosso dominio.
    """
    print("\n[INFO] Fase de Fine-Tuning...")
    
    import tensorflow as tf
    
    # Descongelar as ultimas 30 camadas do MobileNetV2
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 30)
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    camadas_treinaveis = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Camadas descongeladas: {camadas_treinaveis}/{len(base_model.layers)}")
    
    # Recompilar com learning rate menor
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo


def treinar(modelo, X_train, y_train, X_val, y_val, fase="Transfer Learning"):
    """Treina o modelo com early stopping e learning rate scheduling."""
    
    import tensorflow as tf
    
    print(f"\n[TREINO] Treinando ({fase})...")
    print(f"   Treino: {len(X_train)} imagens | Validacao: {len(X_val)} imagens")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    historico = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return historico


def avaliar_modelo(modelo, X_test, y_test, nomes_test=None):
    """Avalia o modelo e mostra metricas detalhadas."""
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "=" * 50)
    print("AVALIACAO DO MODELO")
    print("=" * 50)
    
    # Predicoes
    y_pred_proba = modelo.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Relatorio de classificacao
    print("\nRelatorio de Classificacao:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
    
    # Matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusao:")
    print(f"   {'':>15} | Pred: {CLASSES[0]:<12} | Pred: {CLASSES[1]:<12}")
    print(f"   {'-' * 55}")
    for i, classe in enumerate(CLASSES):
        print(f"   Real: {classe:<12} | {cm[i][0]:<18} | {cm[i][1]:<12}")
    
    # Predicoes individuais
    if nomes_test is not None:
        print("\nPredicoes Individuais:")
        for nome, real, pred, prob in zip(nomes_test, y_test, y_pred, y_pred_proba):
            classe_real = CLASSES[real]
            classe_pred = CLASSES[pred]
            status = "[OK]" if real == pred else "[ERRO]"
            print(f"   {status} {nome:<30} | Real: {classe_real:<15} | "
                  f"Pred: {classe_pred:<15} | Confianca: {max(prob, 1-prob):.1%}")
    
    # Acuracia geral
    acuracia = np.mean(y_pred == y_test)
    print(f"\nAcuracia Geral: {acuracia:.1%}")
    
    return y_pred, y_pred_proba


def plotar_historico(historicos, nome_arquivo="treinamento_grafico.png"):
    """Gera graficos de loss e accuracy durante o treinamento."""
    
    print("\n[INFO] Gerando graficos de treinamento...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Historico de Treinamento - Classificador de Abacaxi", 
                 fontsize=14, fontweight='bold')
    
    cores = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    # Combinar historicos
    loss_total = []
    val_loss_total = []
    acc_total = []
    val_acc_total = []
    
    for h in historicos:
        loss_total.extend(h.history['loss'])
        val_loss_total.extend(h.history['val_loss'])
        acc_total.extend(h.history['accuracy'])
        val_acc_total.extend(h.history['val_accuracy'])
    
    epocas = range(1, len(loss_total) + 1)
    
    # Grafico de Loss
    axes[0].plot(epocas, loss_total, color=cores[0], linewidth=2, label='Treino')
    axes[0].plot(epocas, val_loss_total, color=cores[1], linewidth=2, label='Validacao')
    axes[0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Grafico de Accuracy
    axes[1].plot(epocas, acc_total, color=cores[2], linewidth=2, label='Treino')
    axes[1].plot(epocas, val_acc_total, color=cores[3], linewidth=2, label='Validacao')
    axes[1].set_title('Acuracia', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Acuracia')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    caminho = os.path.join(BASE_DIR, nome_arquivo)
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Grafico salvo em: {nome_arquivo}")


def salvar_modelo(modelo):
    """Salva o modelo treinado."""
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Salvar em formato Keras nativo
    caminho_keras = os.path.join(MODEL_DIR, "classificador_abacaxi.keras")
    modelo.save(caminho_keras)
    print(f"\n[SALVO] Modelo salvo em: {caminho_keras}")
    
    # Salvar informacoes das classes
    info_path = os.path.join(MODEL_DIR, "classes.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        for idx, classe in enumerate(CLASSES):
            f.write(f"{idx}: {classe}\n")
    print(f"   Classes salvas em: {info_path}")
    
    return caminho_keras


def predizer_imagem(modelo, caminho_imagem):
    """Faz predicao para uma unica imagem."""
    
    img = Image.open(caminho_imagem).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = modelo.predict(img_array, verbose=0)[0][0]
    classe_idx = 1 if pred > 0.5 else 0
    classe_nome = CLASSES[classe_idx]
    confianca = pred if classe_idx == 1 else 1 - pred
    
    return classe_nome, confianca


def testar_todas_imagens(modelo):
    """Testa o modelo com todas as imagens do dataset."""
    
    print("\n" + "=" * 60)
    print("TESTE FINAL - Predicoes em Todas as Imagens")
    print("=" * 60)
    
    for classe in CLASSES:
        pasta = os.path.join(DATASET_DIR, classe)
        if not os.path.exists(pasta):
            continue
            
        arquivos = sorted([f for f in os.listdir(pasta) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
        
        print(f"\nClasse: {classe}")
        print("-" * 50)
        
        for arquivo in arquivos:
            caminho = os.path.join(pasta, arquivo)
            classe_pred, confianca = predizer_imagem(modelo, caminho)
            
            status = "[OK]" if classe_pred == classe else "[ERRO]"
            print(f"   {status} {arquivo:<20} -> {classe_pred:<15} "
                  f"(confianca: {confianca:.1%})")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Pipeline completo de treinamento."""
    
    print("=" * 60)
    print("CLASSIFICADOR DE FOLHAGEM DE ABACAXI")
    print("   Folhagem (com/sem fruto) vs Fruto Sozinho")
    print("=" * 60)
    
    # 1. Verificar dataset
    contagem = verificar_dataset()
    
    # 2. Carregar imagens
    X, y, nomes = carregar_imagens()
    
    # 3. Data Augmentation
    X_aug, y_aug = criar_augmentacoes(X, y)
    
    # 4. Separar treino e validacao
    from sklearn.model_selection import train_test_split
    
    # Usar as imagens originais para validacao, augmentadas para treino
    # Primeiro, separar algumas originais para validacao
    indices_originais = list(range(len(X)))
    
    # Estratificado: garante que ambas as classes estao em treino e validacao
    try:
        idx_train, idx_val = train_test_split(
            indices_originais, test_size=0.25, 
            stratify=y, random_state=42
        )
    except ValueError:
        # Se uma classe tem poucas imagens, usar split simples
        np.random.seed(42)
        np.random.shuffle(indices_originais)
        split = max(1, len(indices_originais) // 4)
        idx_val = indices_originais[:split]
        idx_train = indices_originais[split:]
    
    X_val = X[idx_val]
    y_val = y[idx_val]
    nomes_val = [nomes[i] for i in idx_val]
    
    # Para treino, usar augmentacao apenas das imagens de treino
    X_train_orig = X[idx_train]
    y_train_orig = y[idx_train]
    X_train, y_train = criar_augmentacoes(X_train_orig, y_train_orig)
    
    # Embaralhar treino
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    print(f"\nDivisao do Dataset:")
    print(f"   Treino:    {len(X_train)} imagens (augmentadas)")
    print(f"   Validacao: {len(X_val)} imagens (originais)")
    
    # 5. Construir modelo
    modelo, base_model = construir_modelo()
    
    # 6. Fase 1: Transfer Learning (camadas base congeladas)
    historico1 = treinar(modelo, X_train, y_train, X_val, y_val, 
                         fase="Transfer Learning")
    
    # 7. Fase 2: Fine-Tuning (descongelar ultimas camadas)
    modelo = fine_tune_modelo(modelo, base_model)
    historico2 = treinar(modelo, X_train, y_train, X_val, y_val,
                         fase="Fine-Tuning")
    
    # 8. Avaliar modelo
    avaliar_modelo(modelo, X_val, y_val, nomes_val)
    
    # 9. Graficos de treinamento
    plotar_historico([historico1, historico2])
    
    # 10. Salvar modelo
    salvar_modelo(modelo)
    
    # 11. Testar com todas as imagens
    testar_todas_imagens(modelo)
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUIDO!")
    print("=" * 60)
    print(f"\nModelo salvo em: modelo/classificador_abacaxi.keras")
    print(f"Grafico salvo em: treinamento_grafico.png")
    print(f"\nPara usar o modelo treinado em novas imagens:")
    print(f"   from train_model import predizer_imagem")
    print(f"   import tensorflow as tf")
    print(f"   modelo = tf.keras.models.load_model('modelo/classificador_abacaxi.keras')")
    print(f"   classe, confianca = predizer_imagem(modelo, 'caminho/para/imagem.png')")


if __name__ == "__main__":
    main()
