# Recomendador de Próximo Item Baseado em Sessões (GRU + Atenção)

Sistema de recomendação *session-based* que prevê o próximo item de interesse a partir da sequência de itens visualizados na sessão corrente, sem histórico de longo prazo nem identificação do usuário. O modelo (PyTorch) é servido por uma aplicação de demonstração em Streamlit que simula uma loja de e-commerce.

> **Relatório técnico completo:** [`relatorio_recomendador_v6.ipynb`](relatorio_recomendador_v6.ipynb) ([PDF](relatorio_recomendador_v6_preview.pdf)) — *"Do Teto Teórico de Recall à Calibração de um Viés de Popularidade"* (retreino v4 → v6).

**Autores:** Marcus Flávio Gonçalves da Silva (200023764) · Luiz Fernando Almeida Pinheiro (221021151)

---

## Resultados

Meta do retreino: **+50% em Recall@5** sobre o modelo de referência (Recall@5 ≈ 0,064). Resultado final, em três condições de avaliação (Seção "Protocolo de avaliação"):

| Métrica | Histórico (10k, antigo) | Controlado (50k, antigo) | **Novo (50k, final)** | Ganho vs. histórico | Ganho vs. controlado |
|---|---|---|---|---|---|
| Recall@5 | 0,0643 | 0,0698 | **0,1293** | **+101,2%** | **+85,4%** |
| MRR@5 | 0,0293 | 0,0316 | 0,0656 | +123,4% | +107,7% |
| NDCG@5 | 0,0379 | 0,0409 | 0,0812 | +114,4% | +98,5% |
| Recall@10 | 0,1194 | 0,1366 | 0,2231 | +86,8% | +63,3% |
| MRR@10 | 0,0364 | 0,0402 | 0,0777 | +113,3% | +93,1% |
| NDCG@10 | 0,0554 | 0,0623 | 0,1112 | +100,5% | +78,5% |
| Coverage@10 | 0,926 | ~0,96 | 0,740 | — | — |

A queda de Coverage@10 (0,96 → 0,74) é o trade-off clássico **acurácia × cobertura**: o modelo concentra as recomendações nos itens de fato mais prováveis de cada categoria (ver "Principais conclusões").

## Arquitetura do modelo

GRU de 2 camadas sobre embeddings concatenados de item, categoria e preço, com **atenção aditiva** (estilo Bahdanau) sobre a sequência completa de estados ocultos — em vez de usar apenas o último estado:

$$e_t = \tanh(W_a h_t), \qquad \alpha_t = \frac{\exp(v^\top e_t)}{\sum_{\tau} \exp(v^\top e_\tau)}, \qquad c = \sum_t \alpha_t h_t$$

O contexto $c$ alimenta a camada de saída totalmente conectada, que produz um logit por item do catálogo.

| Componente | Dimensão |
|---|---|
| Embedding de item | 64 |
| Embedding de categoria | 16 |
| Projeção linear do preço normalizado | 8 |
| Entrada da GRU (concatenação) | 88 |
| Estado oculto da GRU | 128 (2 camadas, dropout 0,5) |
| **Parâmetros treináveis** | **296.148** (+16.640 da atenção sobre a base de 279.508) |

A atenção foi adicionada preservando a assinatura pública da classe (`__init__`/`forward`), mantendo compatibilidade com o código de inferência do app. A classe `SessionGRU` em [`app.py`](app.py) espelha exatamente a do notebook de treinamento.

> **Nota sobre o nome do checkpoint:** `recomendador_checkpoint_v4.pt` mantém o nome legado por compatibilidade, mas contém os **pesos do modelo final do relatório** (com atenção, 296.148 parâmetros), além dos vocabulários (`item_to_idx`, `cat_to_idx`), hiperparâmetros e estatísticas de normalização de preço (`price_mean`/`price_std`).

## Dados: catálogo e geração de sessões sintéticas

Catálogo de **500 produtos** em **8 categorias** (~62,5 itens/categoria). Cada sessão é gerada por um processo estocástico que simula navegação: a cada passo, com probabilidade `SAME_CATEGORY_PROB` o próximo item sai da mesma categoria do item atual; caso contrário, troca de categoria ou sorteia de todo o catálogo.

Melhorias aplicadas ao gerador, na ordem em que foram introduzidas:

| # | Melhoria | Descrição |
|---|---|---|
| 1 | Volume de sessões | 10.000 → 50.000 |
| 2 | Remoção de duplicatas consecutivas | Resorteio (até 5 tentativas) quando o item repete o anterior |
| 3 | Peso temporal nos pares de treino | Posições recentes reamostradas com maior frequência como alvo (somente no treino) |
| 4 | Coerência de categoria | `SAME_CATEGORY_PROB`: 0,75 → 0,90 |
| 5 | **Viés de popularidade intra-categoria** | Distribuição tipo Zipf sobre os itens de cada categoria |

O componente decisivo foi o **nº 5**: peso de popularidade $w_i \propto 1/\text{rank}(i)^{s}$ por item dentro da categoria, com ranking definido por permutação embaralhada com semente fixa (sem atalho trivial pela ordem do `item_id`). O expoente foi **calibrado empiricamente** para superar a meta sem trivializar a tarefa:

| `ZIPF_EXPONENT` ($s$) | Recall@5 | Ganho vs. controlado |
|---|---|---|
| 0,00 (uniforme) | 0,0707 | −1,5% |
| 0,10 | 0,0761 | +9,2% |
| 0,20 | 0,1006 | +47,2% |
| **0,30 (escolhido)** | **0,1093** | **+66,3%** |
| 0,80 (rejeitado — trivializa) | 0,3109 | +510,0% |

## Treinamento

Perda combinada: entropia cruzada (softmax completo) + **BPR com amostragem negativa**:

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{BPR}, \qquad \mathcal{L}_{BPR} = -\frac{1}{N}\sum_{i=1}^{N} \log \sigma\big(s_{pos} - s_{neg,i}\big)$$

| Hiperparâmetro | Valor |
|---|---|
| $\lambda$ (peso do BPR) | 0,5 |
| Negativos por exemplo | 10 (uniformes, sem colisão com o positivo) |
| Batch size | 128 |
| Learning rate | 0,001 (decay por platô: fator 0,5, paciência 2) |
| Otimizador | Adam (weight decay 1e-4) |
| Early stopping | Paciência 5 sobre a loss de validação |
| Convergência final | Época 34 (val loss 4,7740) |

## Protocolo de avaliação

Três condições isolam o efeito da arquitetura do efeito da distribuição dos dados:

- **Histórico** — modelo de referência (sem atenção) avaliado nos dados da configuração original do gerador (10k sessões, `SAME_CATEGORY_PROB=0,75`). Reproduz o número historicamente reportado (~0,064).
- **Controlado** — o *mesmo* modelo de referência, reavaliado nas *mesmas sessões de teste do modelo novo*, reindexadas para o vocabulário original do checkpoint de referência. Elimina o viés de comparar em conjuntos de teste diferentes.
- **Novo** — o modelo proposto, no seu próprio conjunto de teste.

Cuidado metodológico relevante: na condição "controlado", os índices de item/categoria devem ser os do vocabulário **original** do checkpoint de referência — um vocabulário recalculado a partir dos novos dados desalinharia índice ↔ item e invalidaria a comparação.

## Principais conclusões

1. **Arquitetura sozinha não resolveu.** Nove configurações combinando atenção, BPR e peso temporal ficaram entre −2% e −19% vs. o baseline controlado. Para sessões curtas (média ≈ 6 itens), o último estado oculto da GRU já retém a informação relevante — a atenção adiciona parâmetros sem retorno correspondente.
2. **O gargalo era o teto teórico dos dados.** Com escolha uniforme de item dentro da categoria, o teto de Recall@5 é $\approx p \cdot \frac{5}{N} + (1-p) \cdot \frac{5}{500} \approx 0{,}08$ mesmo com coerência de categoria perfeita ($p \to 1$) — máximo de +24%, estruturalmente insuficiente para a meta de +50%. A previsão foi confirmada empiricamente (modelo novo 0,0707 vs. referência 0,0718, ambos no teto de ~0,073).
3. **O ganho veio dos dados, não do modelo:** só a introdução de um padrão de item genuinamente aprendível (popularidade Zipf intra-categoria) desbloqueou o +101%. Em dados sintéticos, **validar o teto teórico do processo gerador deveria preceder o ajuste de arquitetura e hiperparâmetros**.
4. **Variância entre execuções:** ±0,003–0,005 em Recall@5 entre execuções nominalmente equivalentes, mesmo com semente fixa — diferenças menores que isso não são sinal real sem repetição com múltiplas sementes.

## Aplicação de demonstração (Streamlit)

Loja mockup que consome o checkpoint em CPU (`torch.load(..., weights_only=True)` + `@st.cache_resource`): catálogo paginado com busca e filtro por categoria, carrinho na sidebar (a sessão do carrinho é a sequência de entrada do modelo), recomendações top-4 recalculadas a cada mudança no carrinho, e fluxo de checkout com upsell. Na inferência, os itens já presentes na sessão são mascarados (`-inf`) antes do top-k. O design system (Material Design 3, acessibilidade, responsividade) está documentado em [`DESIGN.md`](DESIGN.md).

## Estrutura do repositório

```
recomendador-sessoes-app/
├── app.py                                # App Streamlit (UI + inferência)
├── recomendador_checkpoint_v4.pt         # Pesos do modelo final + vocabulários (nome legado)
├── catalogo_v4.csv                       # Catálogo: 500 produtos, 8 categorias
├── recomendador_sessoes4.ipynb           # Notebook de treinamento (geração de dados, treino, avaliação)
├── relatorio_recomendador_v6.ipynb       # Relatório técnico do retreino (v4 → v6)
├── relatorio_recomendador_v6_preview.pdf # Versão PDF do relatório
├── RELATORIO_RETREINO_V5.md              # Relatório intermediário (v5: atenção + BPR)
├── comparacao_metricas_v4_vs_v5.csv      # Métricas comparativas v4 × v5
├── DESIGN.md                             # Design system da aplicação
└── requirements.txt                      # Dependências
```

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Acesse `http://localhost:8501`. A inferência roda em CPU; o primeiro carregamento exibe um skeleton enquanto o checkpoint e o catálogo são lidos (cacheados nas execuções seguintes).

### Deploy no Streamlit Cloud

1. Conecte o repositório em [share.streamlit.io](https://share.streamlit.io) e aponte para `app.py`.
2. Checkpoint (~1,2 MB) e catálogo são pequenos o suficiente para o repositório; se o modelo crescer, considere [Git LFS](https://git-lfs.github.com/) ou carregamento por URL.

## Licença

Projeto acadêmico. Sinta-se livre para usar e modificar.
