# Classificador de Folhagem de Abacaxi 🍍

Este projeto utiliza Machine Learning (Deep Learning) para identificar a folhagem do abacaxi em imagens aéreas, diferenciando plantas (com ou sem fruto) de imagens que contêm apenas o fruto isolado (colhido).

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter o Python instalado (recomendado 3.10 ou superior). Instale as dependências necessárias:

```bash
pip install tensorflow Pillow numpy matplotlib scikit-learn
```

### 2. Organização do Dataset

Coloque suas imagens brutas na pasta `images/`. Em seguida, execute o script de organização para estruturar os dados em pastas de classes (`dataset/folhagem` e `dataset/fruto_sozinho`):

```bash
python organize_dataset.py
```

*Nota: O script classifica as imagens atuais como 'folhagem'. Para a classe 'fruto_sozinho', você deve adicionar imagens de abacaxis colhidos na pasta correspondente.*

### 3. Treinamento do Modelo

Para treinar o modelo utilizando Transfer Learning (MobileNetV2) e Data Augmentation, execute:

```bash
python train_model.py
```

O script fará o seguinte:
- Carregará as imagens e aplicará aumentos aleatórios (rotação, brilho, etc).
- Treinará o modelo em duas fases (Transfer Learning e Fine-Tuning).
- Salvará o modelo final em `modelo/classificador_abacaxi.keras`.
- Gerará um gráfico de desempenho em `treinamento_grafico.png`.

### 4. Utilizando o Modelo (Predição)

Você pode usar o modelo treinado em novos scripts da seguinte forma:

```python
import tensorflow as tf
from train_model import predizer_imagem

# Carregar o modelo treinado
modelo = tf.keras.models.load_model('modelo/classificador_abacaxi.keras')

# Fazer a predição de uma imagem
caminho = 'caminho/para/sua/imagem.png'
classe, confianca = predizer_imagem(modelo, caminho)

print(f"Resultado: {classe} ({confianca:.1%})")
```

## 📂 Estrutura do Projeto

- `images/`: Pasta com as imagens originais da folhagem.
- `dataset/`: Imagens organizadas por classe para o treinamento.
- `modelo/`: Contém o modelo treinado e a lista de classes.
- `organize_dataset.py`: Script para preparar a estrutura de pastas.
- `train_model.py`: Script principal de treinamento e avaliação.
- `treinamento_grafico.png`: Visualização da precisão e perda durante o treino.

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem principal.
- **TensorFlow/Keras**: Framework de Deep Learning.
- **MobileNetV2**: Arquitetura de rede neural pré-treinada.
- **Pillow**: Processamento de imagens.
- **Matplotlib**: Geração de gráficos.
- **Scikit-learn**: Métricas de avaliação.
