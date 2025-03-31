# Detecção de Objetos com YOLOv8

## Visão Geral
Este projeto utiliza o modelo YOLOv8, desenvolvido pela Ultralytics, para treinar um modelo personalizado de detecção de objetos com base em um dataset de 166 imagens anotadas, contendo as classes "carro" (classe 0) e "bike" (classe 1). O objetivo é treinar o modelo para detectar esses objetos em imagens e realizar análise em tempo real em vídeos.

## Objetivos
- Treinar um modelo YOLOv8 com um dataset personalizado.
- Realizar detecção de objetos em tempo real em vídeos.
- Documentar o processo para facilitar a replicação e manutenção.

## Requisitos

### Sistema Operacional
- Linux (testado em ambiente Bash 5.2).

### Hardware
- CPU (ex.: AMD Ryzen 5 5500U) ou GPU (recomendado para desempenho otimizado).

### Software
- Python 3.8 ou superior.
- PyTorch 1.8 ou superior.
- Pacote Ultralytics (versão 8.3.99 ou superior).

## Estrutura do Projeto

```
/train-yolo/
├── train.py              # Script para treinar o modelo
├── detect.py             # Script para detecção em tempo real
├── dataset/
│   ├── dataset.yaml      # Configuração do dataset
│   ├── train/
│   │   ├── images/       # Imagens de treinamento (~132 imagens)
│   │   └── labels/       # Anotações de treinamento (~132 arquivos .txt)
│   └── val/
│       ├── images/       # Imagens de validação (~34 imagens)
│       └── labels/       # Anotações de validação (~34 arquivos .txt)
└── runs/                 # Diretório para resultados do treinamento
    └── detect/
        └── trainX/       # Pesos do modelo (ex.: best.pt)
```

## Descrição dos Arquivos
- `train.py`: Script Python para treinar o modelo YOLOv8 com o dataset especificado.
- `detect.py`: Script Python para realizar detecção em tempo real em vídeos usando o modelo treinado.
- `split_dataset.py`: Script auxiliar para dividir o dataset em conjuntos de treino (80%) e validação (20%).
- `dataset.yaml`: Arquivo de configuração do dataset, contendo os caminhos para as imagens e os nomes das classes.

## Configuração do Ambiente

### Instalação das Dependências
Instale o Python 3.8 ou superior.

Instale o PyTorch:
```bash
pip install torch torchvision
```
Escolha a versão adequada ao seu hardware (CPU ou GPU) em PyTorch.

Instale ou atualize o pacote Ultralytics:
```bash
pip install -U ultralytics
```

### Verificação
Verifique as versões instaladas:
```bash
python -c "import torch; import ultralytics; print(torch.__version__, ultralytics.__version__)"
```
Saída esperada: algo como `2.6.0+cu124 8.3.99`.

## Preparação do Dataset

### Estrutura do Dataset
O dataset contém 166 imagens com anotações no formato YOLO (.txt com linhas no formato `classe x_centro y_centro largura altura`). As classes são:

- `0`: "carro"
- `1`: "bike"

### Divisão do Dataset
Divida o dataset em:

- **Treino**: 80% (~132 imagens).
- **Validação**: 20% (~34 imagens).

### Execução
```bash
python train.py
```

## Detecção em Tempo Real

### Execução
```bash
python detect.py --video /caminho/para/seu/video.mp4
```

## Considerações Adicionais
- **Ajuste de Parâmetros**: Monitore o `mAP` para evitar overfitting.
- **Hardware**: Use uma GPU para melhor desempenho.
- **Aumento de Dados**: Habilite `mosaic=1.0` no treinamento.

## Recursos
- [Documentação Ultralytics](https://docs.ultralytics.com/)
- [Repositório GitHub Ultralytics](https://github.com/ultralytics/yolov8)
