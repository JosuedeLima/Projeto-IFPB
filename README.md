# Robô de adubação em plantações de abacaxi — visão computacional

> **Projeto de Pesquisa — Instituto Federal da Paraíba (IFPB)**  
> Detecção de objetos em tempo real aplicada à agricultura de precisão.

---

## Visão geral

Este repositório contém o módulo de **visão computacional** de um robô destinado à **adubação seletiva**: o veículo percorre fileiras da plantação e, ao identificar pé de abacaxi com **olho do fruto** visível, posiciona o bico dosador sobre o **alvo** definido pela detecção do `olho_abacaxi`.

A base do detector é **YOLOv8** via [Ultralytics](https://docs.ultralytics.com/), com transfer learning a partir dos pesos pré-treinados no COCO.

**Melhorias recentes em relação à versão inicial**

- Fluxo único recomendado de dataset: [`prepare_dataset_robust.py`](prepare_dataset_robust.py) (limpeza de labels, split **train/val** estratificado, relatório de qualidade).
- Validação de anotações rígida em [`prepare_annotations.py`](prepare_annotations.py) (apenas formato **YOLO bbox**: 5 números por linha).
- Treino exige **split de validação real** — não usar treino como `val`.
- Inferência “robô” com **confirmação temporal** do alvo, **estabilidade espacial** do centro detectado e **cooldown** após adubar (reduce falsos acionamentos).
- `.gitignore` orientado à **segurança**: não versionar imagens pesadas, pesos (`.pt`), saídas de treino nem credenciais.

---

## Problema tratado

A adubação correta mira o **“olho”** da planta (região do fruto/apicalidade visível nas imagens capturadas), evitando desperdício e dano à folhagem quando o modelo e a mecânica estiverem calibrados.

---

## Classes detectadas (ordem fixa para YOLO)

| Classe         | ID | Descrição                        | Papel na lógica                          |
|----------------|----|----------------------------------|------------------------------------------|
| `pe_abacaxi`   | 0  | Folhagem / pé da planta         | Gatilho: há planta relevante na cena      |
| `olho_abacaxi` | 1  | Fruto / olho visível na copa    | Alvo: ponto prioritário para o dosador   |

Ao exportar de Roboflow, LabelImg ou CVAT, use **esta ordem de IDs**.

---

## Lógica do robô (inferência em tempo real)

Resumo por frame:

1. Se **não** há detecção de pé → seguir trajetória (`caminhar`).
2. Se há pé **sem** olho aceito pela rede → não adubar; registrar cenário típico de oclusão ou planta sem fruto visível (`aguardar`).
3. Se há pé **e** olho com confiança acima do limiar → **confirmar** o alvo antes de disparar (`aguardar` durante a confirmação).

**Critérios de segurança (constantes em [`train_yolo_abacaxi.py`](train_yolo_abacaxi.py))**

| Parâmetro                         | Valor padrão | Função                                               |
|-----------------------------------|-------------|-------------------------------------------------------|
| `REQUIRED_CONSECUTIVE_FRAMES`     | 3           | Só libera “adubar” após N frames aceitáveis seguidos  |
| `TARGET_STABILITY_RADIUS_PX`      | 40          | Centro do olho deve variar pouco entre frames       |
| `ADUBACAO_COOLDOWN_FRAMES`       | 20          | Evita repetir disparo em sequência no mesmo passe   |
| `CONF_THRESHOLD`                  | 0.50        | Confiança mínima das detecções                       |

Integração física (`_mover_bico`, `_acionar_dosador`) permanece como stub para ROS2/GPIO para integração com hardware robótico.

---

## Arquitetura técnica

| Componente          | Tecnologia                                      |
|---------------------|-------------------------------------------------|
| Detecção            | YOLOv8n (base `yolov8n.pt`, ajustável no script) |
| Framework           | Ultralytics                                     |
| Captura             | OpenCV (`cv2.VideoCapture`)                     |
| Linguagem           | Python 3.10+ recomendado (3.12 no desenvolvimento) |
| Hardware alvo       | PC embarcado / Raspberry Pi + câmera            |

Motivos típicos de YOLOv8: velocidade em hardware modesto, bom desempenho com poucos dados via transfer learning, exportação ONNX/TensorRT para deploy.

---

## Estrutura do repositório

```
projectIFPBComputerVison/
├── train_yolo_abacaxi.py      # Treino, teste estático e modo câmera (+ lógica do robô)
├── prepare_dataset_robust.py # Prepara dataset: limpa labels, split, data.yaml local, relatório
├── prepare_annotations.py    # Valida formato e classes dos .txt
├── requirements.txt
├── README.md
├── .gitignore
├── modelo/                   # Referência opcional para nomes de classe (podem ficar só locais)
└── dataset/
    ├── data.yaml             # Preferir caminho portável antes de publicar (ver abaixo)
    ├── labels_cleaned/       # Saída intermediária do prepare_dataset_robust (opcional/local)
    ├── labels_raw/           # Opcional: coloque aqui exports brutos antes do pipeline robusto
    ├── images/train|val/      # Imagens (normalmente ignoradas pelo Git por tamanho)
    └── labels/train|val/      # Labels versionáveis (*.txt YOLO bbox)
```

O `.gitignore` atual prioriza commit **enxuto**: em geral entram apenas código, configs e **`dataset/**/labels/*.txt`** mais **`dataset/**/*.yaml`**; ficam **fora** imagens grandes, pesos `.pt`, `runs/`, caches e segredos.

---

## Instalação

**Pré-requisito:** Python 3.10+.

```bash
git clone https://github.com/JosuedeLima/Projeto-IFPB.git   # ou a URL atual do projeto
cd Projeto-IFPB

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
```

Para treino acelerado com GPU, instale o PyTorch com CUDA **antes** de instalar ultralytics conforme [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Como usar (fluxo recomendado)

### 1. Imagens brutas

Coloque as fotos na pasta **`images/`** na raiz (ela é ignorada no Git para não pesar no remoto).

### 2. Preparar dataset e split train/val

**Primeira vez (só fotos ainda não anotadas — cria `.txt` vazios onde faltar):**

```bash
python prepare_dataset_robust.py --from-images
```

**Já existe export dos labels (YOLO bbox) junto ao nome das imagens:**

- Coloque os `.txt` de origem em `dataset/labels_raw/` com o **mesmo stem** dos arquivos em `images/`.
- Execute:

```bash
python prepare_dataset_robust.py --images-dir images --labels-dir dataset/labels_raw
```

O script:

- mantém apenas linhas válidas (**5 valores** por objeto: classe, centro x/y, largura e altura, normalizados);
- faz split estratificado (perfis como “só pé”, “pé + olho”, etc.);
- gera/atualiza `dataset/data.yaml`;
- imprime um resumo e grava **`dataset/quality_report.txt`**.

Parâmetros úteis: `--val-ratio 0.2`, `--seed 42`.

### 3. Anotar (fora dos scripts)

Use Roboflow, LabelImg ou CVAT exportando **detecção YOLO**. Exemplo por linha:

```
0 cx cy w h    # pe_abacaxi
1 cx cy w h    # olho_abacaxi
```

### 4. Validar anotações

```bash
python prepare_annotations.py
```

Corrige inconsistências até não haver erro de arquivo ou falta de label por imagem.

### 5. Treinar

```bash
python train_yolo_abacaxi.py --modo treinar
```

Antes do treino, o script regenera **`dataset/data.yaml`** com base na estrutura detectada. É obrigatório existir **validação** em:

- `dataset/images/train` e `dataset/images/val`, ou
- layout alternativo `dataset/train/images` + `dataset/val|valid/images`.

O melhor peso exportado aparece sob `runs/abacaxi_detector/weights/best.pt` e é copiado para `modelo_yolo/detector_abacaxi.pt` (pastas/arquivos esses ficam ignorados no Git quando não devem ir ao remoto).

### 6. Testar uma imagem

```bash
python train_yolo_abacaxi.py --modo testar --imagem caminho\foto.jpg
```

### 7. Modo robô (câmera)

```bash
python train_yolo_abacaxi.py --modo robo
```

Saída opcional na janela OpenCV (`q` encerra).

---


---

## Augmentations usadas no treino (plantação)

Parâmetros principais em `model.train(...)` do script: `flipud`, `fliplr`, `degrees`, `hsv_*`, `mosaic`, etc., pensados para variação de ângulo e iluminação de campo.

---

## Métricas após o treino

Em `runs/abacaxi_detector/` (local, normalmente não versionada): curvas de loss, precision/recall, mAP, matriz de confusão e batches de validação.

---

## Dependências

Ver [`requirements.txt`](requirements.txt): `ultralytics`, `opencv-python`, `Pillow`, `numpy`, `pyyaml`.

---

## Licença

Projeto acadêmico desenvolvido no **Instituto Federal da Paraíba (IFPB)**.

---

*Stack: Python · YOLOv8 (Ultralytics) · OpenCV*
