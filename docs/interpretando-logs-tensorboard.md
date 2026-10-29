# Como interpretar os logs do TensorBoard

Os logs são gerados pelo callback `TensorBoard` durante o treinamento. Cada execução
fica em uma pasta com o nome:

```text
logs/fit/<nome_do_modelo>_<data-hora>/
```

Exemplo:

```text
logs/fit/resnet50v2_20261028-194315/
```

Abra os logs com:

```bash
tensorboard --logdir logs/fit
```

Depois, acesse o endereço exibido no terminal, normalmente
`http://localhost:6006`.

## Identificação da execução

O nome da pasta contém:

- **Nome do modelo:** obtido dinamicamente do backbone usado, por exemplo `resnet50v2`.
- **Data e hora:** no formato `AAAAMMDD-HHMMSS`.
- **Train:** dados usados para atualizar os pesos.
- **Validation:** dados usados apenas para medir a generalização ao fim de cada época.

Uma **época** é uma passagem completa pelo conjunto de treinamento. O eixo horizontal
dos gráficos normalmente representa a época; o campo `step` representa a posição da
medição durante a execução.

## Scalars: métricas numéricas

As métricas devem ser lidas como séries ao longo das épocas, e não como um valor
isolado. O valor da época 1 mostra o início do aprendizado; os valores seguintes
mostram se o modelo continua melhorando, estabiliza ou começa a memorizar o
conjunto de treinamento.

No TensorBoard, selecione **Smoothed** para enxergar a tendência geral e desative
o suavizamento quando precisar verificar o valor exato de uma época. O suavizamento
é apenas visual: não altera o treinamento nem os valores armazenados.

### Resumo rápido

| Métrica | Conjunto | Melhor direção | O que responde |
|---|---|---|---|
| `loss` | treinamento | diminuir | O modelo está aprendendo os exemplos usados para treiná-lo? |
| `val_loss` | validação | diminuir | O modelo generaliza para exemplos não usados no ajuste? |
| `accuracy` | treinamento | aumentar | Qual fração das classes foi prevista corretamente no treino? |
| `val_accuracy` | validação | aumentar | Qual fração foi prevista corretamente em imagens de validação? |
| `learning_rate` | configuração do otimizador | depende da fase | Qual o tamanho dos próximos ajustes dos pesos? |

`loss` e `val_loss` são erros; portanto, valores menores são melhores. `accuracy` e
`val_accuracy` são acertos; portanto, valores maiores são melhores.

### `loss`

É o erro médio calculado no conjunto de treinamento. Neste projeto, é a
`CategoricalCrossentropy` com `label_smoothing=0.1`. Para cada imagem, a rede
produz uma probabilidade para cada uma das 11 classes, e a cross-entropy penaliza
principalmente a probabilidade atribuída à classe correta:

```text
cross-entropy = - soma(alvo_da_classe * log(probabilidade_da_classe))
loss = média da cross-entropy de todas as imagens do lote/época
```

Uma previsão correta e confiante tende a produzir `loss` baixo. Uma previsão
errada e confiante produz `loss` alto. Por isso, `loss` pode piorar bastante mesmo
quando a `accuracy` não muda: a quantidade de acertos pode ser igual, mas a rede
pode ter ficado mais confiante nos erros.

O `label_smoothing=0.1` suaviza o alvo: em vez de exigir probabilidade `1.0` para
a classe correta e `0.0` para as demais, distribui uma pequena parte da certeza
entre as outras classes. Isso reduz excesso de confiança e significa que o valor
de `loss` não deve ser comparado diretamente com uma execução que não use o mesmo
`label_smoothing`.

- Menor valor geralmente é melhor.
- Deve ser analisado junto com `val_loss`.
- Não representa diretamente uma porcentagem.
- Pode aumentar quando o modelo fica mais confiante em previsões erradas.

### `val_loss`

É a mesma função de erro calculada no conjunto de validação, sem atualizar os pesos.
Ela mede o comportamento em imagens que não foram usadas para calcular os ajustes
dos pesos naquela época. Por isso, normalmente é mais importante que `loss` para
escolher o modelo que será usado depois.

Neste projeto, `val_loss` é usada para:

- salvar o melhor checkpoint (`ModelCheckpoint`);
- reduzir a taxa de aprendizado (`ReduceLROnPlateau`);
- interromper o treinamento antecipadamente (`EarlyStopping`).

O menor `val_loss` é o principal critério para escolher o modelo deste treinamento.
Uma diferença pequena entre épocas pode ser apenas variação dos dados; observe a
tendência e não apenas uma oscilação de um ponto.

O `EarlyStopping` está configurado com `min_delta=0.01`. Portanto, uma redução
menor que `0.01` não é considerada uma melhoria significativa para a paciência do
callback, mesmo que o número exibido tenha diminuído um pouco.

### `accuracy`

É a acurácia categórica no conjunto de treinamento. Uma imagem conta como correta
somente quando a classe com maior probabilidade (`argmax`) é a classe esperada:

```text
accuracy = número de previsões corretas / número total de imagens
```

O valor fica entre `0` e `1`. Para convertê-lo em porcentagem, multiplique por `100`.
Por exemplo, `0.642653` corresponde a aproximadamente `64,27%`.

Essa métrica não informa o grau de confiança nem o desempenho de cada classe:
uma previsão com probabilidade `0.51` e outra com `0.99` contam igualmente como
um acerto. Também pode esconder classes com poucos exemplos; por isso, deve ser
complementada por métricas por classe e matriz de confusão.

### `val_accuracy`

É a mesma contagem de acertos, mas no conjunto de validação. É o indicador mais
intuitivo para estimar o comportamento em imagens não usadas para atualizar os
pesos. Como ela usa `argmax`, não precisa somar exatamente `1.0` às probabilidades
para ser interpretada; o que importa é a classe com maior probabilidade.

Compare `accuracy` e `val_accuracy`:

- valores próximos sugerem melhor generalização;
- `accuracy` muito maior que `val_accuracy` pode indicar overfitting;
- ambas baixas podem indicar underfitting, dados difíceis, classes desbalanceadas
  ou necessidade de revisar o treinamento.

O intervalo entre as duas pode ser calculado como:

```text
gap = accuracy - val_accuracy
```

Por exemplo, `accuracy=0.80` e `val_accuracy=0.60` produzem `gap=0.20`, ou
20 pontos percentuais. Um gap crescente junto com `loss` caindo e `val_loss`
subindo é um forte sinal de overfitting. Um gap pequeno não garante qualidade:
as duas métricas podem estar igualmente baixas.

Como o projeto possui 11 classes, uma referência ingênua de escolha aleatória
uniforme seria aproximadamente `1/11 = 0.0909` (9,09%). Essa referência não
substitui um baseline real, pois a distribuição das classes pode ser desigual.

### Taxa de aprendizado

Se uma série `learning_rate` aparecer nos logs, ela representa o tamanho máximo
dos ajustes calculados pelo otimizador Adam. Não é uma taxa de acerto e não deve
ser comparada com `accuracy`:

- fase 1: `0.001` (`base_lr`);
- fase 2: `0.00002` (`fine_tune_lr`);
- o `ReduceLROnPlateau` pode reduzir esse valor quando `val_loss` deixa de melhorar;
- o limite inferior configurado é `0.000001`.

Uma taxa muito alta pode causar oscilações em `loss` e `val_loss`; uma taxa muito
baixa pode tornar o treinamento lento ou impedir melhorias perceptíveis. A mudança
entre as duas fases é intencional: o backbone começa congelado e depois é ajustado
com passos menores para preservar as representações pré-treinadas.

### Relação entre os padrões das curvas

| `loss` | `val_loss` | `accuracy` / `val_accuracy` | Interpretação provável |
|---|---|---|---|
| cai | cai | sobem | aprendizado saudável e melhora de generalização |
| cai | sobe | treino sobe, validação para ou cai | overfitting; considere o checkpoint da menor `val_loss` |
| permanece alto | permanece alto | ambas baixas | underfitting, dados difíceis, pré-processamento ou configuração inadequada |
| oscila muito | oscila muito | oscila | taxa de aprendizado alta, lote pequeno ou dados muito variados |
| cai pouco | cai pouco | quase não mudam | aprendizado estagnado; verifique taxa de aprendizado e qualidade dos dados |

Esses padrões são sinais de diagnóstico, não provas isoladas. Confirme a hipótese
com uma avaliação no conjunto de teste e com resultados por classe.

## Histograms e Distributions: parâmetros das camadas

O projeto usa `histogram_freq=1`, portanto o TensorBoard registra a distribuição dos
pesos nas épocas. Esses gráficos não são métricas de qualidade: mostram como os
parâmetros internos estão distribuídos e mudando.

### `kernel`

É a matriz principal de pesos de uma camada:

- em convoluções, são filtros que detectam padrões locais;
- em camadas densas, conectam entradas às saídas;
- valores próximos de zero indicam pouca contribuição individual;
- distribuições que crescem muito podem indicar instabilidade ou gradientes
  excessivos;
- uma distribuição sem variação por muitas épocas pode indicar que a camada está
  congelada ou recebendo pouca atualização.

No início, o backbone `ResNet50V2` é congelado. As camadas do backbone só passam a
ser ajustadas na fase de fine-tuning, com as primeiras 100 camadas permanecendo
congeladas.

### `bias`

É o deslocamento adicionado à saída de uma camada. Ele permite que a camada ajuste
seu ponto de ativação independentemente dos valores de entrada. Assim como `kernel`,
deve ser observado como distribuição, não como uma métrica isolada.

### `gamma`

É o fator de escala (`scale`) das camadas Batch Normalization:

```text
saída = gamma * valor_normalizado + beta
```

- `gamma` próximo de `1` mantém a escala normalizada;
- `gamma` próximo de `0` reduz a contribuição daquele canal;
- valores muito grandes podem indicar que a camada está compensando uma
  normalização excessiva.

### `beta`

É o deslocamento (`offset`) aprendido pela Batch Normalization:

- `beta` próximo de `0` indica pouco deslocamento;
- valores positivos ou negativos deslocam a ativação do canal;
- sua interpretação deve ser feita junto com `gamma` e com as métricas de validação.

### Estatísticas da Batch Normalization

Se aparecerem `moving_mean` e `moving_variance`, eles são estatísticas acumuladas
usadas pela Batch Normalization durante a inferência:

- `moving_mean`: média móvel das ativações;
- `moving_variance`: variância móvel das ativações;
- não são pesos treináveis diretamente;
- mudanças abruptas podem indicar instabilidade ou lotes pouco representativos.

## Graph

A aba **Graphs** mostra o grafo computacional do modelo. Ela ajuda a verificar a
sequência de operações, o backbone, o `GlobalAveragePooling2D`, o `Dropout` e a
camada final `Dense`.

O grafo não mostra se o modelo está bom. Para avaliar qualidade, use `loss`,
`val_loss`, `accuracy` e `val_accuracy`.

## Como avaliar uma execução

1. Verifique se `loss` e `val_loss` diminuem no início.
2. Localize a época de menor `val_loss`.
3. Compare `accuracy` com `val_accuracy` nessa época.
4. Verifique se `val_loss` começa a subir enquanto `loss` continua caindo; isso
   é um sinal comum de overfitting.
5. Observe os histogramas para identificar pesos explodindo, parados ou concentrados
   em valores extremos.
6. Use o checkpoint correspondente à menor `val_loss`, que é o arquivo `best_*.keras`.

## Relação com o arquivo `.txt`

O arquivo em `logs/infos/` resume a execução. Ele contém:

- nome e data do treinamento;
- caminho do modelo final;
- caminho do melhor modelo;
- quantidade de épocas executadas;
- época de menor `val_loss`;
- métricas da melhor época e da última época;
- quantidade total de parâmetros;
- classes utilizadas.

As métricas do modelo final salvo devem ser interpretadas principalmente pela seção
**“métricas do modelo final salvo (melhor época)”**, pois o treinamento usa
`restore_best_weights=True`. As métricas da última época mostram apenas o estado
observado no fim da execução.

## Limitações de interpretação

- Uma única métrica não é suficiente para avaliar o modelo.
- `accuracy` pode esconder desempenho ruim em classes minoritárias.
- `loss` e `val_loss` só podem ser comparados diretamente entre execuções que usam
  a mesma função de perda e configuração.
- Histogramas de `kernel`, `beta` e `gamma` ajudam a diagnosticar o treinamento,
  mas não substituem métricas por classe, matriz de confusão ou avaliação em um
  conjunto de teste separado.
