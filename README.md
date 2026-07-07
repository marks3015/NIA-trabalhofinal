# Recomendador de Próximo Item Baseado em Sessões (GRU + Atenção)

Sistema de recomendação *session-based* que prevê o próximo item de interesse a partir da sequência de produtos visualizados na sessão corrente — sem depender de histórico de longo prazo nem de identificação do usuário. O modelo é uma rede neural híbrida (PyTorch) servida por uma aplicação Streamlit que simula uma loja de e-commerce.

> **Relatório técnico completo:** [`relatorio_recomendador_v6.ipynb`](relatorio_recomendador_v6.ipynb) ([PDF](relatorio_recomendador_v6_preview.pdf)) — *"Do Teto Teórico de Recall à Calibração de um Viés de Popularidade"* (retreino v4 → v6).

**Autores:** Marcus Flávio Gonçalves da Silva (200023764) · Luiz Fernando Almeida Pinheiro (221021151)

---

## 1. Motivação e contexto

Em e-commerce, grande parte da navegação acontece sem login ou com usuários anônimos. Nesses cenários, não é possível usar histórico de compras passadas ou perfis demográficos. A recomendação precisa ser feita **apenas com o que o usuário clicou na sessão atual**.

Este projeto explora essa tarefa sobre um catálogo sintético de 500 produtos em 8 categorias. O objetivo acadêmico foi melhorar o Recall@5 do sistema em pelo menos **50%** sobre um modelo de referência (Recall@5 ≈ 0,064). O caminho até a meta revelou uma lição metodológica importante: **a arquitetura só aprende padrões que existem nos dados**.

### Por que essa abordagem?

- **GRU (Gated Recurrent Unit):** captura a ordem e a dependência entre os itens da sessão de forma eficiente, sendo um padrão clássico da literatura (GRU4Rec, Hidasi et al.).
- **Atenção aditiva (Bahdanau-style):** permite que o modelo pondere todos os itens da sessão, não apenas o último — útil quando itens anteriores também são informativos.
- **Embedding de categoria e preço:** o modelo recebe mais contexto do que apenas o ID do item, aproximando-se de dados reais de catálogo.
- **Loss combinada (CrossEntropy + BPR):** a entropia cruzada modela a distribuição completa; o BPR reforça a ordenação relativa entre o item positivo e itens negativos, alinhando-se às métricas de ranking.

### Por que dados sintéticos?

Dados reais de e-commerce são difíceis de obter por questões de privacidade e propriedade. O gerador sintético permite:

1. **Controlar o processo gerador** e entender exatamente o que o modelo pode aprender.
2. **Reproduzir resultados** com sementes fixas.
3. **Testar intervenções** de forma rápida e mensurável.

A desvantagem é que o sinal a ser aprendido precisa ser *propositalmente inserido* — e foi exatamente isso que descobrimos ao longo do trabalho.

---

## 2. Aplicações

Embora o projeto seja acadêmico, a arquitetura e o pipeline podem ser aplicados em diversos cenários reais:

- **Lojas virtuais anônimas:** recomendar "quem viu X também viu Y" para visitantes não logados.
- **Portais de conteúdo:** sugerir o próximo artigo, vídeo ou notícia com base na sequência de consumo da sessão.
- **Apps de delivery e turismo:** recomendar próximo restaurante, hotel ou experiência a partir do que o usuário está explorando no momento.
- **Chatbots e assistentes de compra:** usar a conversa corrente como sessão para sugerir produtos.
- **Benchmarks e pesquisa:** o gerador sintético serve como ambiente controlado para testar novas arquiteturas de recomendação sequencial.

A aplicação Streamlit demonstra o uso em uma loja mockup: o usuário navega pelo catálogo, adiciona itens ao carrinho e recebe recomendações recalculadas a cada novo item.

---

## 3. Arquitetura do modelo

A rede combina três fontes de informação em cada passo da sequência:

| Componente | Dimensão | Motivo |
|---|---|---|
| Embedding de item | 64 | Representação densa do produto no catálogo. |
| Embedding de categoria | 16 | Informação semântica que agrupa produtos similares. |
| Projeção linear do preço normalizado | 8 | Sinal numérico contínuo de valor do produto. |
| **Entrada da GRU** | **88** | Concatenação das três representações acima. |
| Estado oculto da GRU | 128 (2 camadas, dropout 0,5) | Memória sequencial da sessão. |
| Atenção aditiva | +16.640 parâmetros | Pondera todos os estados ocultos da sessão. |
| **Parâmetros treináveis** | **296.148** | — |

A atenção é calculada sobre todos os estados ocultos:

$$e_t = \tanh(W_a h_t), \qquad \alpha_t = \frac{\exp(v^\top e_t)}{\sum_{\tau} \exp(v^\top e_\tau)}, \qquad c = \sum_t \alpha_t h_t$$

O contexto $c$ alimenta a camada linear de saída, que produz um score para cada um dos 500 itens do catálogo.

### Como o modelo relaciona os itens

O modelo não "conhece" os itens por seus nomes ou IDs brutos. Em vez disso, ele aprende **representações densas** (embeddings) que posicionam itens similares próximos no espaço vetorial:

1. **Embedding de item (64 dimensões):** cada um dos 500 produtos é mapeado para um vetor treinável. Itens que aparecem em contextos semelhantes (mesma categoria, preço próximo, padrões de transição parecidos) acabam com embeddings parecidos.

2. **Embedding de categoria (16 dimensões):** a categoria de cada item também vira um vetor. Isso ajuda o modelo a generalizar: se ele aprendeu que "Fone de Ouvido" frequentemente precede "Carregador Portátil" na categoria Eletrônicos, o embedding de categoria reforça esse agrupamento.

3. **Preço normalizado (8 dimensões):** o preço passa por uma projeção linear após normalização (`z = (preço - média) / desvio`). Isso dá ao modelo um sinal contínuo de valor, permitindo que ele aprenda, por exemplo, que itens de faixa de preço similar tendem a ser vistos em sequência.

4. **GRU de 2 camadas:** a sequência de vetores de entrada (88 dimensões) passa pela GRU, que mantém um **estado oculto** atualizado a cada passo. Esse estado funciona como uma "memória" da sessão: ele condensa o que o usuário viu até agora e na ordem em que viu.

5. **Atenção aditiva:** em vez de usar apenas o último estado oculto, o modelo calcula um peso de importância para cada passo da sessão e monta um contexto ponderado. Isso permite que um item visto há alguns cliques ainda influencie a recomendação, se o modelo julgar relevante.

6. **Camada de saída:** o contexto final é multiplicado por uma matriz que projeta de volta para o espaço dos 500 itens. O resultado é um score para cada produto; aplicamos softmax para obter probabilidades e ranqueamos os itens.

Em resumo: o modelo relaciona itens aprendendo **similaridades de embedding** e **padrões de transição** a partir das sessões de treino. Quanto mais vezes dois produtos aparecem em sequência (ou na mesma categoria com popularidade similar), mais próximos seus embeddings tendem a ficar.

> **Compatibilidade:** a atenção foi adicionada preservando a assinatura pública da classe `SessionGRU` (`__init__`/`forward`), mantendo o `app.py` e o checkpoint funcionais sem reescrita.
> **Nota sobre o nome do checkpoint:** `recomendador_checkpoint_v4.pt` mantém o nome legado, mas contém os pesos do modelo final (296.148 parâmetros, com atenção), além dos vocabulários, hiperparâmetros e estatísticas de normalização de preço.

---

## 4. Dados: catálogo e geração de sessões sintéticas

### Catálogo

- **500 produtos** fictícios
- **8 categorias:** Roupas, Eletrônicos, Alimentos e Bebidas, Casa e Decoração, Beleza e Cuidados, Brinquedos, Esportes, Livros e Papelaria
- Média de ~62,5 itens por categoria

Cada produto possui nome, categoria, preço e URL de imagem. O catálogo é determinístico (mesma semente), o que garante reprodutibilidade.

### Processo gerador de sessões

Cada sessão simula a navegação de um usuário. A cada passo:

- Com probabilidade `SAME_CATEGORY_PROB`, o próximo item é sorteado dentro da mesma categoria do item atual.
- Caso contrário, a sessão troca para outra categoria aleatória ou sorteia um item de todo o catálogo.

As melhorias aplicadas ao longo do trabalho foram:

| # | Melhoria | Descrição | Motivo |
|---|---|---|---|
| 1 | Volume de sessões | 10.000 → 50.000 | Mais dados para reduzir variância e melhorar generalização. |
| 2 | Remoção de duplicatas consecutivas | Resorteio quando o item repete o anterior | Evita transições espúrias do tipo `[A, A]`, irreais em navegação. |
| 3 | Peso temporal nos pares de treino | Posições recentes reamostradas como alvo | Reflete a intuição de que itens mais recentes são mais preditivos (somente no treino). |
| 4 | Coerência de categoria | `SAME_CATEGORY_PROB`: 0,75 → 0,90 | Reduz o ruído de troca aleatória de categoria, tornando o sinal de categoria mais forte. |
| 5 | **Viés de popularidade intra-categoria** | Distribuição tipo Zipf por categoria | Introduz um padrão *dentro* da categoria, não apenas *entre* categorias. |

### O componente decisivo: popularidade Zipf

Inicialmente, a escolha do item dentro de uma categoria era **uniforme**. Isso impunha um **teto teórico de Recall@5 ≈ 0,08** mesmo com coerência de categoria perfeita, porque o modelo só podia acertar "a categoria certa", mas não "o item certo" dentro dela. Com 62,5 itens por categoria e 5 recomendações, o melhor caso teórico era limitado.

Para superar esse teto, introduzimos pesos de popularidade tipo Zipf dentro de cada categoria:

$$w_i \propto \frac{1}{\text{rank}(i)^s}$$

O ranking é definido por uma permutação embaralhada com semente fixa, evitando qualquer atalho pela ordem do `item_id`. O expoente $s$ (`ZIPF_EXPONENT`) foi calibrado empiricamente:

| `ZIPF_EXPONENT` ($s$) | Recall@5 | Ganho vs. controlado | Avaliação |
|---|---|---|---|
| 0,00 (uniforme) | 0,0707 | −1,5% | Teto teórico insuficiente. |
| 0,10 | 0,0761 | +9,2% | Ainda abaixo da meta. |
| 0,20 | 0,1006 | +47,2% | Próximo da meta. |
| **0,30 (escolhido)** | **0,1093** | **+66,3%** | Meta batida sem trivializar. |
| 0,80 (rejeitado) | 0,3109 | +510,0% | Tarefa fica trivial — só memoriza frequência. |

A escolha de $s = 0,30$ equilibra **ganho mensurável** e **tarefa desafiadora**: o modelo ainda precisa entender a sessão, não apenas repetir os itens mais populares.

### Como os dados foram gerados (passo a passo)

O pipeline de geração é determinístico (sementes fixas) e reproduzível. Ele acontece em cinco etapas principais:

#### 1. Geração do catálogo

```python
catalogo = gerar_catalogo_mockup(n=500, seed=42)
```

Para cada um dos 500 `item_id`s:
- Sorteia uma das 8 categorias.
- Sorteia um nome base típico da categoria (ex.: "Fone de Ouvido Sem Fio", "Camiseta Polo") e um adjetivo (ex.: "Smart", "Pro", "Prime").
- Gera um preço aleatório uniforme entre R$ 20,00 e R$ 1.000,00.
- Cria uma URL de imagem placeholder.

O resultado é salvo em `catalogo_v4.csv`. O catálogo não mudou entre versões do modelo — apenas o comportamento de navegação simulado mudou.

#### 2. Construção dos pesos de popularidade (Zipf)

```python
pop_weights = build_category_popularity_weights(catalogo, seed=42)
```

Para cada categoria:
- Pega a lista de itens daquela categoria.
- Embaralha essa lista com uma semente fixa (`random.Random(seed).shuffle`).
- Atribui pesos decrescentes segundo a lei de Zipf: `peso ∝ 1 / (rank + 1)^s`, com `s = 0,30`.
- Normaliza para que a soma dos pesos da categoria seja 1.

O embaralhamento é essencial: se os itens mais populares fossem sempre os de `item_id` menor, o modelo poderia aprender um atalho artificial em vez do padrão real.

#### 3. Geração das 50.000 sessões

```python
sessions = generate_sessions(catalogo, num_sessions=50000,
                             min_len=2, max_len=10, seed=42)
```

Para cada sessão:
- Sorteia um comprimento entre 2 e 10 itens.
- Escolhe uma categoria inicial aleatória.
- Para cada novo item da sessão:
  - Com probabilidade `SAME_CATEGORY_PROB = 0,90`, sorteia um item **da mesma categoria** usando os pesos Zipf.
  - Com probabilidade 0,10, faz uma transição fora da categoria:
    - 50% das vezes: troca para outra categoria aleatória e sorteia um item lá (com pesos Zipf).
    - 50% das vezes: sorteia um item completamente aleatório de todo o catálogo.
  - Se o item sorteado for igual ao anterior, repete o sorteio até 5 vezes, garantindo **zero duplicatas consecutivas** no conjunto final.

Esse processo produz sessões curtas (comprimento médio ≈ 6 itens), realistas para e-commerce, com forte coerência de categoria e viés de popularidade dentro dela.

#### 4. Criação dos pares de treino/validação/teste

```python
# Validação e teste: todos os pares (prefixo, target) sem viés
val_pairs  = [p for s in val_sessions  for p in session_to_pairs(s)]
test_pairs = [p for s in test_sessions for p in session_to_pairs(s)]

# Treino: oversampling de posições recentes
train_pairs = [p for s in train_sessions for p in session_to_pairs_temporal(s)]
```

- Cada sessão de comprimento $L$ gera $L-1$ exemplos de treino: o prefixo `[i₁, ..., i_{t-1}]` tenta prever o target `i_t`.
- O conjunto de treino aplica **peso temporal**: posições mais próximas do fim da sessão são repetidas até `1 + TEMPORAL_BOOST` vezes, aumentando sua frequência relativa sem descartar nenhuma transição real.
- Validação e teste **não usam peso temporal**, para que as métricas reflitam a distribuição real de transições.
- O split é feito **por sessão** (80% treino / 10% validação / 10% teste), evitando vazamento de informação entre conjuntos.

#### 5. Mapeamento para índices do modelo

```python
item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}
idx_to_item = {idx: item for item, idx in item_to_idx.items()}
```

Todos os `item_id`s reais do catálogo são convertidos em índices contíguos usados pelas camadas de embedding. O mapeamento inverso (`idx_to_item`) permite traduzir as previsões do modelo de volta para os produtos do catálogo.

---

## 5. Treinamento

A função de perda combinada une entropia cruzada (softmax completo) com BPR (negative sampling):

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{BPR}, \qquad \mathcal{L}_{BPR} = -\frac{1}{N}\sum_{i=1}^{N} \log \sigma\big(s_{pos} - s_{neg,i}\big)$$

| Hiperparâmetro | Valor | Motivo |
|---|---|---|
| $\lambda$ (peso do BPR) | 0,5 | Reforça o ranking sem abandonar a verossimilhança. |
| Negativos por exemplo | 10 | Amostragem suficiente para contrastar com o positivo. |
| Batch size | 128 | Bom equilíbrio entre velocidade e estabilidade. |
| Learning rate | 0,001 | Taxa padrão para Adam; decay por platô evita estagnação. |
| Otimizador | Adam (weight decay 1e-4) | Regularização L2 leve para evitar overfit. |
| Dropout | 0,5 | Regularização forte dado o catálogo pequeno. |
| Early stopping | Paciência 5 sobre val loss | Evita overfit nos dados sintéticos. |
| Convergência final | Época 34 (val loss 4,7740) | — |

> **Infraestrutura:** todo o treinamento e a avaliação foram executados localmente em um **Mac com chip Apple M5 Silicon** (CPU), usando PyTorch com backend `mps`/`cpu`. O tempo de treinamento do modelo final foi de alguns minutos, beneficiando-se do tamanho reduzido do catálogo (500 itens) e do checkpoint compacto (~1,2 MB).

---

## 6. Resultados

### Protocolo de avaliação

Para isolar o efeito real de cada mudança, comparamos três condições:

- **Histórico:** modelo antigo (sem atenção) avaliado nos dados antigos (10k sessões, `SAME_CATEGORY_PROB=0,75`). Reproduz o número original (~0,064).
- **Controlado:** modelo antigo avaliado nas **mesmas sessões de teste do modelo novo**, reindexadas para o vocabulário do checkpoint antigo. Elimina o viés de comparar distribuições diferentes.
- **Novo:** modelo proposto (atenção + BPR + dados enriquecidos) no seu próprio teste.

### Resultado final

Meta do retreino: **+50% em Recall@5** sobre o modelo de referência.

| Métrica | Histórico (10k, antigo) | Controlado (50k, antigo) | **Novo (50k, final)** | Ganho vs. histórico | Ganho vs. controlado |
|---|---|---|---|---|---|
| Recall@5 | 0,0643 | 0,0698 | **0,1293** | **+101,2%** | **+85,4%** |
| MRR@5 | 0,0293 | 0,0316 | 0,0656 | +123,4% | +107,7% |
| NDCG@5 | 0,0379 | 0,0409 | 0,0812 | +114,4% | +98,5% |
| Recall@10 | 0,1194 | 0,1366 | 0,2231 | +86,8% | +63,3% |
| MRR@10 | 0,0364 | 0,0402 | 0,0777 | +113,3% | +93,1% |
| NDCG@10 | 0,0554 | 0,0623 | 0,1112 | +100,5% | +78,5% |
| Coverage@10 | 0,926 | ~0,96 | 0,740 | — | — |

**A meta foi atingida com folga:** +101% contra o histórico e +85% contra o controlado.

### O que realmente funcionou?

1. **Arquitetura sozinha não resolveu.** Nove configurações combinando atenção, BPR e peso temporal ficaram entre −2% e −19% vs. o baseline controlado. Para sessões curtas (média ≈ 6 itens), o último estado oculto da GRU já retém a informação relevante.
2. **O gargalo era o teto teórico dos dados.** Com escolha uniforme dentro da categoria, o Recall@5 máximo era ≈ 0,08 — insuficiente para +50%.
3. **O ganho veio da estrutura dos dados.** A introdução de um padrão de item genuinamente aprendível (popularidade Zipf intra-categoria) desbloqueou o +101%.
4. **BPR, atenção e peso temporal são boas práticas** que devem ajudar mais em dados reais com padrões sequenciais mais ricos. Neste cenário controlado, o sinal sequencial era fraco comparado ao sinal de popularidade.

### Trade-off: acurácia × cobertura

O Coverage@10 caiu de ~0,96 para 0,74. Isso é esperado: ao concentrar recomendações nos itens mais prováveis de cada categoria, o modelo recomenda menos itens distintos no total. Em aplicações reais, esse trade-off deve ser monitorado para evitar *filter bubbles* excessivos.

---

## 7. Aplicação de demonstração (Streamlit)

A interface simula uma loja de e-commerce completa:

- **Catálogo paginado** com busca por nome e filtro por categoria.
- **Carrinho na sidebar:** cada item adicionado vira parte da sessão de entrada do modelo.
- **Recomendações top-4** recalculadas a cada mudança no carrinho.
- **Checkout com upsell:** carrossel de 12 recomendações para completar o pedido.
- **Design system** baseado no Material Design 3 (documentado em [`DESIGN.md`](DESIGN.md)).

Na inferência, itens já presentes no carrinho são mascarados (`-inf`) antes do top-k, garantindo que o modelo só sugira produtos ainda não vistos.

---

## 8. Estrutura do repositório

```
recomendador-sessoes-app/
├── app.py                                # App Streamlit (UI + inferência)
├── recomendador_checkpoint_v4.pt         # Pesos do modelo final + vocabulários (nome legado)
├── catalogo_v4.csv                       # Catálogo: 500 produtos, 8 categorias
├── recomendador_sessoes4.ipynb           # Notebook de treinamento (geração, treino, avaliação)
├── relatorio_recomendador_v6.ipynb       # Relatório técnico do retreino (v4 → v6)
├── relatorio_recomendador_v6_preview.pdf # Versão PDF do relatório
├── RELATORIO_RETREINO_V5.md              # Relatório intermediário (v5: atenção + BPR)
├── comparacao_metricas_v4_vs_v5.csv      # Métricas comparativas v4 × v5
├── DESIGN.md                             # Design system da aplicação
└── requirements.txt                      # Dependências
```

---

## 9. Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Acesse `http://localhost:8501`. A inferência roda em CPU; o primeiro carregamento exibe um skeleton enquanto o checkpoint e o catálogo são lidos (cacheados nas execuções seguintes).

### Deploy no Streamlit Cloud

1. Conecte o repositório em [share.streamlit.io](https://share.streamlit.io) e aponte para `app.py`.
2. Checkpoint (~1,2 MB) e catálogo são pequenos o suficiente para o repositório; se o modelo crescer, considere [Git LFS](https://git-lfs.github.com/) ou carregamento por URL.

---

## 10. Lições aprendidas e limitações

1. **Valide o teto teórico dos dados antes de ajustar arquitetura.** Em dados sintéticos, o processo gerador impõe um limite superior de desempenho. Ajustar hiperparâmetros não supera esse teto.
2. **Atenção e BPR são neutros quando o sinal sequencial é fraco.** Eles brilham em dados com padrões temporais ricos; aqui, o sinal dominante era de popularidade.
3. **Calibração do viés de popularidade é crítica.** Pouco viés mantém a tarefa impossível; muito viés a torna trivial. A escolha de $s = 0,30$ foi deliberada.
4. **Variância entre execuções:** mesmo com seed fixa, mudanças pequenas de configuração alteram o caminho de treino em ±0,003–0,005 em Recall@5. Diferenças menores que isso não são sinal real sem múltiplas repetições.
5. **Limitação de escopo:** os resultados valem para o catálogo e processo gerador definidos. Em dados reais, é esperado que padrões sequenciais, complementaridade e sazonalidade tornem os componentes de modelagem mais relevantes.

---

## Licença

Projeto acadêmico. Sinta-se livre para usar e modificar.
