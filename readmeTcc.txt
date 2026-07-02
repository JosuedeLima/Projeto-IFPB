================================================================================
  TRABALHO DE CONCLUSÃO DE CURSO (TCC)
  Curso: Informática — Instituto Federal da Paraíba (IFPB)
  Tema: Adubação seletiva de abacaxi com visão computacional e validação web
================================================================================

1. CONTEXTO DO PROJETO
--------------------------------------------------------------------------------

Este projeto faz parte de uma pesquisa aplicada à agricultura de precisão. O
objetivo é desenvolver um sistema de visão computacional capaz de identificar,
em tempo real, duas estruturas da planta de abacaxi:

  - pe_abacaxi   (classe 0): folhagem / pé da planta — indica presença de planta
  - olho_abacaxi (classe 1): fruto visível no topo — alvo da adubação

A detecção é realizada pelo modelo YOLOv8 (Ultralytics), treinado por transfer
learning a partir dos pesos pré-treinados yolov8n.pt (dataset COCO).

O sistema foi pensado para operar em um robô que percorre a plantação e aciona
um dosador de fertilizante sobre o olho do fruto. Como ainda não há orçamento
para construção ou aquisição do robô físico, foi desenvolvido um frontend web
para validar a acurácia do modelo sem hardware embarcado.


2. SITUAÇÃO ANTES DAS MUDANÇAS
--------------------------------------------------------------------------------

Antes desta etapa, o repositório continha apenas scripts Python:

  - train_yolo_abacaxi.py   → treino, teste em imagem estática e modo câmera
                               via janela OpenCV (modo "robô")
  - prepare_dataset_robust.py e prepare_annotations.py → preparação do dataset
  - requirements.txt, README.md e estrutura de labels em dataset/

Limitações da versão anterior:

  a) Não existia interface web acessível por navegador (PC ou celular).
  b) A validação em tempo real dependia de OpenCV no computador local.
  c) Não havia API HTTP para receber frames de dispositivos móveis.
  d) O limiar de confiança do robô (50%) era inadequado para o validador web
     quando o modelo ainda está em fase inicial de treino.
  e) Não havia diagnóstico automático da qualidade do modelo ao iniciar o
     servidor.
  f) Conflitos de porta (erro WinError 10048) não eram tratados com mensagem
     clara ao usuário.


3. MUDANÇAS IMPLEMENTADAS NESTA ETAPA
--------------------------------------------------------------------------------

3.1. FRONTEND WEB (pasta frontend/)
-----------------------------------

Foi criada uma interface responsiva (mobile-first) com os arquivos:

  frontend/
    index.html          → estrutura da página
    css/styles.css      → layout adaptável a celular e desktop
    js/app.js           → lógica da câmera, envio de frames e desenho das caixas

Funcionalidades do frontend:

  [1] Abertura da câmera do dispositivo (webcam do PC ou câmera do celular)
      via API getUserMedia do navegador.

  [2] Captura periódica de frames (~6–10 FPS) e envio ao servidor em JPEG.

  [3] Exibição em tempo real das caixas delimitadoras (bounding boxes) sobre
      o vídeo:
        - Verde  → pe_abacaxi
        - Laranja → olho_abacaxi
      Cada caixa mostra o nome da classe e o percentual de confiança.

  [4] Painel lateral (desktop) ou empilhado (mobile) com:
        - botão Iniciar/Parar câmera
        - botão Trocar câmera (frontal/traseira ou entre webcams)
        - slider de confiança mínima (1% a 95%)
        - lista das detecções do frame atual
        - indicadores de FPS e latência de inferência (ms)

  [5] Modo "Testar imagem": upload de arquivo de imagem (.jpg, .png) para
      validar o modelo sem usar a câmera — útil em ambiente de desenvolvimento
      e para apresentação do TCC.

  [6] Badge de status ("Modelo pronto" / "Servidor offline") e alerta visual
      quando o modelo apresenta confiança baixa nas imagens de teste.

  [7] Layout responsivo com CSS Grid no desktop e coluna única no celular,
      respeitando safe-area para dispositivos com notch.


3.2. SERVIDOR WEB (api_server.py)
---------------------------------

Novo módulo que expõe API REST e serve os arquivos estáticos do frontend.

Endpoints:

  GET  /              → página principal (index.html)
  GET  /static/...    → CSS e JavaScript
  GET  /api/health    → status do modelo, classes e diagnóstico de qualidade
  POST /api/detect    → recebe imagem (multipart/form-data) e retorna JSON
                        com detecções, dimensões do frame e tempo de inferência

Parâmetros do servidor (linha de comando):

  python api_server.py
  python api_server.py --modelo modelo_yolo/detector_abacaxi.pt --porta 8000
  python api_server.py --host 0.0.0.0 --porta 8001

O host 0.0.0.0 permite acesso na rede local (ex.: celular na mesma Wi-Fi).

Tecnologias adicionadas ao requirements.txt:
  - fastapi
  - uvicorn[standard]
  - python-multipart


3.3. AJUSTES NO MÓDULO DE INFERÊNCIA (train_yolo_abacaxi.py)
------------------------------------------------------------

Foi adicionado o método detectar_frame() na classe RoboAdubador:

  - Executa apenas a detecção YOLO, sem a lógica de decisão do robô
    (adubar / aguardar / caminhar).
  - Aceita parâmetros conf (limiar de confiança) e imgsz (tamanho de entrada).
  - Retorna lista de detecções com: classe, classe_id, confiança, bbox e centro.
  - Reutilizado tanto pelo api_server.py quanto pelo modo robô existente.

Separação intencional:
  - Modo robô (produção): CONF_THRESHOLD = 0.50 (50%)
  - Modo validador web:   WEB_CONF_DEFAULT = 0.01 (1%) em api_server.py


4. CORREÇÃO DOS ERROS IDENTIFICADOS
--------------------------------------------------------------------------------

4.1. Erro de porta ocupada (WinError 10048)
-------------------------------------------
Sintoma:
  Ao executar python api_server.py pela segunda vez, o servidor falhava com:
  "error while attempting to bind on address ('0.0.0.0', 8000):
   [winerror 10048] normalmente é permitida apenas uma utilização de cada
   endereço de soquete"

Causa:
  Uma instância anterior do uvicorn ainda estava rodando na porta 8000.

Correção:
  - Função porta_disponivel() verifica a porta antes de iniciar.
  - Mensagem clara orientando fechar o terminal anterior (Ctrl+C) ou usar
    --porta 8001.


4.2. Modelo não detectava / labels não apareciam no frontend
------------------------------------------------------------
Sintoma:
  A câmera abria, o servidor respondia, mas nenhuma caixa era exibida.

Causa raiz (diagnosticada com testes nas imagens em images/):
  O modelo treinado com poucas imagens (~15) apresenta confiança máxima de
  aproximadamente 1,1% nas fotos de treino. O frontend usava limiar padrão
  de 50%, filtrando todas as detecções. O problema não era falha de código,
  e sim incompatibilidade entre o limiar configurado e a maturidade atual
  do modelo.

Correções aplicadas:
  a) Limiar padrão do validador web reduzido para 1% (slider mínimo: 1%).
  b) Diagnóstico automático ao iniciar o servidor (_avaliar_modelo): testa
     uma imagem em images/ e exibe aviso se a confiança máxima for < 25%.
  c) Endpoint /api/health retorna campo "diagnostico" com alerta para o
     frontend exibir banner amarelo ao usuário.
  d) Mensagem na lista de detecções: "Nenhum objeto acima do limiar. Baixe
     a confiança mínima." quando não há resultados.
  e) Modo "Testar imagem" para validar com fotos conhecidas do dataset.
  f) Canvas de overlay com z-index explícito (z-index: 2) para garantir que
     as caixas fiquem visíveis sobre o elemento <video>.
  g) Indicador visual "Analisando…" durante o processamento de cada frame.

Recomendação para o TCC:
  Retreinar o modelo com mais imagens anotadas (ideal: 100+) e, após melhoria
  das métricas (mAP, precisão), elevar o slider de confiança gradualmente
  até o valor de produção (50%).


4.3. Carregamento do modelo via uvicorn direto
----------------------------------------------
Sintoma potencial:
  Se o servidor fosse iniciado com "uvicorn api_server:app" sem passar por
  main(), o modelo poderia não ser carregado (robo = None).

Correção:
  - Handler lifespan no FastAPI carrega o modelo automaticamente na inicialização
    se ainda não estiver carregado.
 
 
+4.4. Reestruturação do Projeto, Docker & Frontend Premium
+-------------------------------------------------------
+Para preparar o projeto para deploy em produção no Render de forma segura:
+  - Separação Backend/Frontend: O repositório foi organizado em pastas isoladas
+    (backend/ e frontend/). O api_server.py foi atualizado para buscar os arquivos
+    do frontend no nível superior (BASE_DIR.parent / "frontend").
+  - Suporte a Docker: Foi adicionado um backend/Dockerfile permitindo rodar a
+    aplicação inteira em containers isolados no Render/Railway.
+  - Frontend Premium: O visual foi reconstruído com um tema escuro e moderno
+    (Outfit font, glassmorphism, pulse animations).
+  - Métrica de Acurácia: Foi adicionado um novo cartão no frontend que busca e
+    exibe a "Melhor Acurácia (Confiança)" em tempo real para cada detecção.
+  - Limpeza de Deprecados: Foram apagados os diretórios obsoletos fotosAntigas/,
+    modelo/ (antigo modelo Keras de classificação) e runs/.


5. ARQUITETURA DO VALIDADOR WEB
--------------------------------------------------------------------------------

  ┌─────────────────┐         POST /api/detect          ┌──────────────────┐
  │   Navegador     │  ──── (frame JPEG + confiança) ──►  │  api_server.py   │
  │  (PC / Celular) │                                     │  (FastAPI)       │
  │                 │  ◄──── JSON {deteccoes, bbox} ────  │                  │
  │  video + canvas │                                     │  RoboAdubador    │
  └─────────────────┘                                     │  + YOLOv8 .pt    │
                                                          └──────────────────┘

Fluxo por frame:
  1. Navegador captura frame da câmera (ou imagem enviada pelo usuário).
  2. Redimensiona para até 640 px de largura e codifica em JPEG.
  3. Envia via POST para /api/detect?conf=0.01.
  4. Servidor decodifica com OpenCV, executa YOLOv8 e retorna coordenadas.
  5. JavaScript desenha as caixas no <canvas> sobreposto ao vídeo.


6. ESTRUTURA ATUAL DO REPOSITÓRIO
--------------------------------------------------------------------------------

projectIFPBComputerVison/
├── backend/                   ← NOVO: códigos e pesos do servidor
│   ├── api_server.py          ← Servidor web e API REST (FastAPI)
│   ├── train_yolo_abacaxi.py  ← Treino, teste e controle do Robô
│   ├── prepare_dataset_robust.py
│   ├── prepare_annotations.py
│   ├── requirements.txt       ← Dependências do backend
│   ├── Dockerfile             ← Configuração de container para deploy
│   ├── yolov8n.pt             ← Pesos base do YOLOv8
│   ├── modelo_yolo/           ← detector_abacaxi.pt (pesos treinados)
│   └── images/                ← Imagens locais para autodiagnóstico
├── frontend/                  ← Interface do validador (HTML/CSS/JS)
│   ├── index.html             ← Layout e badge de Melhor Acurácia
│   ├── css/styles.css         ← Estilização Premium (Glassmorphism & Gradients)
│   └── js/app.js              ← Lógica de captura, inferência e desenho de bboxes
├── dataset/                   ← Configurações e labels estruturadas de treino/val
├── README.md                  ← Seção do validador web
└── readmeTcc.txt              ← ESTE ARQUIVO (Documentação do TCC)


7. COMO EXECUTAR O VALIDADOR (PARA DEMONSTRAÇÃO DO TCC)
--------------------------------------------------------------------------------

Pré-requisitos:
  - Python 3.10 ou superior
  - Modelo treinado em backend/modelo_yolo/detector_abacaxi.pt
  - Instalar dependências: pip install -r backend/requirements.txt

Execução Local:

  1. Entre na pasta backend/:
       cd backend

  2. Iniciar o servidor web:
       python api_server.py

  3. Abrir no navegador:
       http://127.0.0.1:8000

Execução via Docker (Produção):

  1. Construa a imagem a partir da raiz do projeto:
       docker build -f backend/Dockerfile -t abacaxi-detector .

  2. Execute o container:
       docker run -p 8000:8000 abacaxi-detector

  4. Para testar no celular (mesma rede Wi-Fi):
       http://<IP-do-computador>:8000
       (descobrir IP: ipconfig no Windows)

  5. Para demonstração sem câmera:
       Clicar em "Testar imagem" e selecionar uma foto de images/
       Manter confiança em 1% até o modelo ser retreinado.

  6. Se a porta 8000 estiver ocupada:
       python api_server.py --porta 8001


8. CLASSES E LÓGICA DO ROBÔ (REFERÊNCIA PARA O TCC)
--------------------------------------------------------------------------------

| Classe         | ID | Função na adubação                          |
|----------------|----|---------------------------------------------|
| pe_abacaxi     | 0  | Gatilho: há planta relevante na cena        |
| olho_abacaxi   | 1  | Alvo: ponto para posicionar o dosador       |

Lógica do robô (modo produção — train_yolo_abacaxi.py --modo robo):
  - Sem pé detectado        → caminhar
  - Pé sem olho visível     → aguardar (não adubar)
  - Pé + olho confirmados   → adubar (após 3 frames estáveis + cooldown)

O validador web NÃO executa essa lógica de adubação; exibe apenas as
detecções brutas do YOLO para avaliação visual da acurácia.


9. LIMITAÇÕES CONHECIDAS E TRABALHOS FUTUROS
--------------------------------------------------------------------------------

Limitações atuais:
  - Dataset pequeno (~15 imagens): modelo com baixa confiança e generalização
    limitada; métricas de treino (mAP ~0,52) indicam necessidade de mais dados.
  - Inferência no servidor (PC): latência depende do hardware; em CPU pode
    ficar entre 50 ms e 500 ms por frame.
  - Câmera no celular via HTTP na rede local pode exigir HTTPS em alguns
    navegadores (restrição de segurança do getUserMedia).
  - Integração física com robô (GPIO, ROS2) permanece como stub no código.

Trabalhos futuros sugeridos para continuidade do TCC ou pós-graduação:
  - Ampliar dataset com fotos de campo (diferentes ângulos, iluminação, estágios
    da planta).
  - Retreinar YOLOv8 e comparar variantes (yolov8n vs yolov8s).
  - Exportar modelo para ONNX/TensorRT para deploy em Raspberry Pi ou Jetson.
  - Implementar WebSocket para streaming com menor latência.
  - Integrar validador web com log de sessão (precisão por frame, export CSV).
  - Construir/adaptar hardware do robô e calibrar homografia pixel → coordenada
    física do dosador.


10. REFERÊNCIAS TÉCNICAS
--------------------------------------------------------------------------------

  - Ultralytics YOLOv8: https://docs.ultralytics.com/
  - FastAPI: https://fastapi.tiangolo.com/
  - OpenCV: https://opencv.org/
  - API MediaDevices (câmera no navegador): MDN Web Docs — getUserMedia


================================================================================
  Documento gerado para apoio à redação do TCC — IFPB, Curso de Informática.
  Projeto: Visão computacional aplicada à adubação seletiva de abacaxi.
  Stack: Python · YOLOv8 · FastAPI · HTML/CSS/JavaScript · OpenCV
================================================================================
