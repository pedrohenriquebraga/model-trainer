# Treinador de modelos de visão

Este diretório contém o pipeline de treinamento de um modelo de visão computacional
com TensorFlow/Keras. O pipeline lê imagens organizadas em diretórios por classe,
faz aumento de dados, treina um classificador baseado e pode exportar o modelo treinado para TensorFlow Lite
(`.tflite`).

## Estrutura

```text
trainer/
├── augmentations.py   # Aumentos de dados e pré-processamento
├── build.py           # Montagem, treinamento e salvamento do modelo
├── build.sh           # Atalho para executar o treinamento
├── convert.py         # Conversão de .keras para TensorFlow Lite
├── convert.sh         # Atalho para conversão float16/builtin
├── requirements.txt   # Dependências Python
├── data/
│   ├── train/<classe>/ # Imagens usadas no treinamento
│   ├── val/<classe>/   # Imagens usadas na validação
│   └── test/<classe>/  # Reservado para avaliação externa
├── models/             # Modelos .keras e .tflite gerados
├── logs/               # Logs do TensorBoard e execuções anteriores
└── reports/            # Relatórios e matrizes de confusão, quando gerados
```

## Requisitos

- Python 3.12 (o ambiente atualmente usado pelo projeto é Python 3.12);
- pip e ambiente virtual;
- GPU NVIDIA compatível é opcional. Sem GPU, o TensorFlow executa na CPU, mas o
  treinamento pode ser consideravelmente mais lento;

As versões principais estão fixadas em [requirements.txt](./requirements.txt).
O pacote `tensorflow[and-cuda]` é usado para instalar o TensorFlow e, quando
suportado pela plataforma, os componentes CUDA necessários.

## Instalação

Na raiz deste diretório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Windows, ative o ambiente com:

```powershell
.\.venv\Scripts\Activate.ps1
```

Para verificar a instalação:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## Preparação do dataset

O carregador `tf.keras.preprocessing.image_dataset_from_directory` espera uma
subpasta para cada classe:

```text
data/
├── train/
│   ├── bed/
│   │   ├── imagem-001.jpg
│   │   └── imagem-002.jpg
│   ├── cat/
│   └── ...
├── val/
│   ├── bed/
│   ├── cat/
│   └── ...
└── test/
    ├── bed/
    ├── cat/
    └── ...
```

Recomendações:

1. Use as mesmas classes em `train` e `val`. O script combina os nomes
   encontrados nos dois diretórios, mas uma classe sem exemplos em `train` não
   pode ser aprendida.
2. Separe as imagens antes do treinamento, evitando que cópias ou imagens quase
   idênticas apareçam em mais de uma divisão.
3. Use imagens RGB ou formatos que o TensorFlow consiga ler.
4. Mantenha quantidade razoavelmente equilibrada de exemplos por classe. Se
   houver grande desbalanceamento, considere pesos de classe ou amostragem
   apropriada.
5. O diretório `test` não é usado por `build.py`; ele deve ser reservado para
   uma avaliação independente.

Ao iniciar, o treinamento gera [classes.txt](./classes.txt) com uma classe por
linha. A ordem é alfabética e deve ser preservada no código de inferência para
interpretar corretamente os índices da saída `softmax`.

## Treinamento

Execute o atalho:

```bash
./build.sh
```

Ou execute diretamente:

```bash
.venv/bin/python build.py
```

O [build.sh](./build.sh) configura o `LD_LIBRARY_PATH` com bibliotecas NVIDIA
instaladas no ambiente virtual antes de chamar o Python. Se o treinamento for
executado diretamente e a GPU não for reconhecida, prefira o script ou configure
as bibliotecas CUDA conforme a instalação local.

### Fluxo do treinamento

O [build.py](./build.py) executa três fases:

1. **Cabeça do classificador:** MobileNetV2 congelada por 10 épocas; somente a
   camada `Dense` final é treinada.
2. **Fine-tuning inicial:** descongela as últimas 30 camadas treináveis do
   backbone, mantendo camadas `BatchNormalization` congeladas, por até mais
   20 épocas.
3. **Fine-tuning ampliado:** permite até 100 camadas finais do backbone e usa
   uma taxa de aprendizado menor até o limite total de 100 épocas (ou até o modelo não apresentar mais evolução).

Também são aplicados:

- redimensionamento com preenchimento para o tamanho definido no `img_size`;
- aumentos aleatórios de rotação, escala, espelhamento, cor, translação,
  cisalhamento, desfoque/nitidez, recorte, cutout e ruído, somente no treino;
- `CategoricalCrossentropy` com label smoothing de `0.1`;
- `Dropout(0.4)` e regularização L2 na camada classificadora;
- `EarlyStopping`, `ReduceLROnPlateau`, checkpoint do melhor `val_loss` e
  TensorBoard;

Os modelos são salvos em `models/best_<data>-<hora>.keras` e os logs em
`logs/fit/<data>-<hora>`. O checkpoint monitora `val_loss`, não a acurácia.

Para acompanhar uma execução:

```bash
tensorboard --logdir logs
```

## Exportação para TensorFlow Lite

Depois de gerar um `.keras`, execute:

```bash
./convert.sh
```

Esse atalho usa o último modelo `.keras` encontrado em `models/` quando o
caminho padrão não existe e tenta exportar com quantização `float16` e runtime
`builtin`.

Para escolher explicitamente o modelo e o destino:

```bash
.venv/bin/python convert.py \
  --input models/best_20261028-084554.keras \
  --output models/modelo.tflite \
  --quantization float16 \
  --runtime builtin
```

Opções de quantização:

- `float16`: reduz o tamanho usando pesos float16;
- `dynamic`: otimização dinâmica, sem dataset representativo;
- `int8`: quantização inteira, usando até 150 imagens de `data/train` como
  dataset representativo e definindo entrada/saída como `int8`;
- `float`: conversão sem quantização.

O conversor detecta automaticamente a resolução e o número de canais do
modelo. `--runtime builtin` é preferível para dispositivos que não possuem
Flex delegate; `--runtime flex` ou `--runtime auto` pode ser necessário quando
alguma operação do modelo não está disponível entre as operações TFLite
nativas.

## Treinar outro modelo

O pipeline não recebe o nome do backbone pela linha de comando. Para treinar
uma arquitetura diferente, faça as alterações abaixo em [build.py](./build.py).

### 1. Trocar o backbone

Altere o import e a construção de `base_model`. Por exemplo, para usar outra
rede disponível em `keras.applications`:

```python
from keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3),
)
```

Mantenha `include_top=False`, pois a cabeça final é criada pelo próprio
pipeline. Nem toda arquitetura aceita exatamente os mesmos argumentos; consulte
a assinatura da aplicação escolhida.

### 2. Ajustar o pré-processamento

O modelo atual recebe pixels em `float32` no intervalo `0..255` e o pipeline
usa `resize_with_pad`. Cada backbone pode exigir um pré-processamento diferente:

- use a função `preprocess_input` da arquitetura, se ela exigir normalização
  específica;
- ajuste [augmentations.py](./augmentations.py) para que os aumentos ocorram
  antes da normalização;
- confirme se a resolução mínima e a proporção de aspecto são compatíveis;
- mantenha o mesmo pré-processamento na inferência e na conversão TFLite.

MobileNetV2, por exemplo, normalmente usa valores em `[-1, 1]` quando seu
`preprocess_input` é aplicado. O código atual não chama essa função
explicitamente; essa decisão deve ser revista ao trocar de arquitetura.

### 3. Recalibrar o fine-tuning

As quantidades `initial_fine_tune_layers` e `fine_tune_layers` dependem do
número e da organização das camadas do backbone. Ao trocar de modelo:

- verifique `len(base_model.layers)`;
- reduza os valores se a rede tiver poucas camadas;
- mantenha `BatchNormalization` congelada em datasets pequenos ou instáveis;
- diminua `fine_tune_lr_initial` e `fine_tune_lr` se houver overfitting ou
  oscilações no `val_loss`.

Backbones maiores também podem exigir redução de `batch_size` para caber na
memória da GPU.

### 4. Ajustar a cabeça e os hiperparâmetros

Para alterar a capacidade do classificador, ajuste `Dropout`, `classifier_l2`
e a camada `Dense`. O número de saídas (`num_classes`) é calculado
automaticamente a partir dos diretórios, portanto não deve ser fixado
manualmente. Para mudanças mais profundas, preserve a ativação `softmax` e
`CategoricalCrossentropy` enquanto o problema continuar sendo multiclasse com
uma classe por imagem.

Revise também:

```python
batch_size
img_size
epochs
epochs_frozen
epochs_fine_tune_initial
base_lr
fine_tune_lr_initial
fine_tune_lr
```

Após cada alteração, treine novamente, compare `val_loss` e `accuracy`, e
avalie o conjunto `test` separadamente. Um `val_loss` menor não substitui uma
avaliação por classe, matriz de confusão e inspeção de erros.

### 5. Validar a conversão

Arquiteturas diferentes podem usar operações que não são suportadas pelo
runtime `builtin`. Se `./convert.sh` falhar:

1. tente `--runtime auto`;
2. tente `--quantization dynamic` ou `--quantization float`;
3. use `--runtime flex` somente se o ambiente de destino oferecer Flex;
4. valide a saída do `.tflite` no mesmo formato de entrada do modelo Keras.

Para modelos com entradas ou saídas diferentes, atualize também o código de
inferência no projeto consumidor. O arquivo `classes.txt` deve acompanhar o
modelo exportado.

## Solução de problemas

- **Nenhuma classe encontrada:** verifique se `data/train` existe e contém
  subdiretórios com imagens.
- **Classe ausente ou ordem diferente:** gere o `classes.txt` e distribua esse
  arquivo junto com o modelo.
- **Sem GPU:** confirme a instalação do driver/CUDA; o script continua podendo
  executar na CPU.
- **Out of memory:** reduza `batch_size`, `img_size` ou use um backbone menor.
- **Overfitting:** aumente os dados, intensifique ou ajuste os aumentos, eleve
  `Dropout`/L2 ou reduza o fine-tuning.
- **TFLite não converte:** tente os modos e runtimes descritos na seção de
  exportação e verifique se o modelo não contém operações incompatíveis.
