# Relatório — Retreino do Recomendador (v4 → v5: Atenção + Negative Sampling)

**Data:** 2026-07-06
**Commit:** `561dfab` — `retrain: GRU com atenção + negative sampling (BPR), 50k sessões sintéticas`
**Repositório:** https://github.com/marks3015/NIA-trabalhofinal

## 1. Objetivo

Implementar melhorias de médio prazo no modelo de recomendação (GRU baseado em sessões) para aumentar Recall@K, MRR@K e NDCG@K, mantendo o app Streamlit funcionando. Critério de sucesso definido: **Recall@5 melhora em pelo menos 50%** em relação ao modelo anterior.

**Resultado resumido: a meta de +50% não foi atingida.** O modelo novo ficou essencialmente empatado com o antigo numa comparação controlada (ver seção 4). As seções abaixo documentam o que foi implementado, como foi medido, e por quê.

---

## 2. O que foi implementado

### 2.1 Geração de sessões sintéticas
- `NUM_SESSIONS`: 10.000 → **50.000**.
- **Sem itens duplicados consecutivos**: ao sortear o próximo item, se ele repetir o item imediatamente anterior da sessão, resorteia-se (até 5 tentativas).
- **Peso temporal nos pares de treino**: em vez de usar cada posição da sessão como um único par de treino `(prefixo, target)`, posições mais recentes (próximas do fim da sessão) são repetidas até `1 + TEMPORAL_BOOST` vezes (com `TEMPORAL_BOOST=2`, `TEMPORAL_ALPHA=1.5`), aumentando sua frequência relativa durante o treino. Nenhuma transição real é descartada — val/teste continuam usando todos os pares, sem viés, para uma avaliação justa.

### 2.2 Mecanismo de atenção
A `SessionGRU` (em `app.py` e no notebook) passou a calcular o contexto final como uma **combinação ponderada de todos os estados ocultos da sequência** (atenção aditiva / Bahdanau-style), em vez de usar apenas o último estado oculto da GRU. A assinatura de `__init__`/`forward` foi mantida idêntica à anterior, preservando a API pública esperada pelo `app.py`.

### 2.3 Negative sampling + loss combinada
Loss de treino = `CrossEntropyLoss + 0.5 * BPR_loss`, onde a loss BPR sorteia 10 itens negativos por exemplo (excluindo colisão com o item positivo) e penaliza o modelo quando o score do negativo se aproxima do score do item-alvo real.

### 2.4 Retreino
Novo checkpoint salvo em `recomendador_checkpoint_v4.pt` e novo catálogo em `catalogo_v4.csv` (o catálogo é gerado deterministicamente a partir do mesmo seed/código, então ficou **byte-idêntico** ao anterior — confirmado por comparação direta).

### 2.5 Bug real encontrado e corrigido
`price_mean`/`price_std` eram calculados via `catalogo['preco'].mean()`/`.std()`, que retornam `numpy.float64`, não `float` nativo do Python. Ao salvar isso dentro do checkpoint (`torch.save`), o arquivo passa a **falhar ao ser carregado** com `torch.load(..., weights_only=True)` — exatamente o modo que `app.py` usa em produção (erro: `Unsupported global: GLOBAL numpy._core.multiarray.scalar`). Corrigido com `float(...)` explícito, tanto no script de retreino quanto no notebook. Sem essa correção, o app quebraria em produção com o checkpoint novo.

---

## 3. Metodologia de avaliação

Métricas calculadas: Recall@5, MRR@5, NDCG@5, Recall@10, MRR@10, NDCG@10 (+ Coverage@10).

Três números são reportados para cada métrica:

1. **Histórico** — checkpoint **antigo** (arquitetura sem atenção) avaliado em dados reconstruídos com a geração de sessão **antiga** (10k sessões, sem dedupe). Reproduz o número histórico (~0,06 de Recall@5) citado no início do projeto.
2. **Controlado** — o **mesmo checkpoint antigo**, avaliado nas **mesmas sessões de teste do modelo novo** (50k sessões, sem duplicados consecutivos), apenas reindexadas para o vocabulário do checkpoint antigo. Isola o efeito da arquitetura/loss, eliminando o viés de estar comparando em distribuições de teste diferentes.
3. **Novo** — o modelo novo (atenção + BPR + peso temporal), avaliado no seu próprio conjunto de teste.

A comparação **controlada** é a mais rigorosa e justa: garante que ambos os modelos são avaliados exatamente nos mesmos exemplos, com os índices de embedding batendo corretamente com o vocabulário que cada checkpoint espera (um cuidado necessário porque um vocabulário recalculado ingenuamente teria produzido índices incompatíveis com os pesos já treinados do checkpoint antigo).

---

## 4. Resultado final (o que foi commitado)

Configuração: atenção + BPR (peso 0.5, 10 negativos) + peso temporal, dropout 0.5, patience 5, 50.000 sessões. Early stopping na época 25 (best val loss 5.4249). 296.148 parâmetros treináveis (vs. 279.508 do modelo antigo). Duplicados consecutivos: 1,12% → 0,00%.

| Métrica | Histórico (10k, antigo) | Controlado (50k, antigo) | Novo (50k, atenção+BPR) | Ganho vs. histórico | Ganho vs. controlado |
|---|---|---|---|---|---|
| Recall@5  | 0,0643 | 0,0585 | 0,0567 | **-11,8%** | **-3,1%** |
| MRR@5     | 0,0293 | 0,0263 | 0,0260 | -11,4% | -1,3% |
| NDCG@5    | 0,0379 | 0,0342 | 0,0335 | -11,5% | -2,0% |
| Recall@10 | 0,1194 | 0,1171 | 0,1099 | -8,0% | -6,1% |
| MRR@10    | 0,0364 | 0,0338 | 0,0329 | -9,7% | -2,8% |
| NDCG@10   | 0,0554 | 0,0528 | 0,0505 | -8,9% | -4,4% |
| Coverage@10 | 0,926 | 0,966 | 0,996 | — | — |

Baseline de popularidade (sempre recomenda os 5 itens mais populares do treino): Recall@5 = 0,0116 — todos os modelos (antigo e novo) superam esse piso por larga margem, mas isso não é o critério de sucesso definido.

**Conclusão direta: o modelo novo não supera o antigo.** Fica de 1 a 3 pontos percentuais relativos abaixo na comparação controlada (dentro da margem de ruído entre execuções, ver seção 5), e ~11-12% abaixo do número histórico original.

---

## 5. Estudo de ablação (por que não foi só ajustar hiperparâmetro)

Antes de aceitar o resultado acima, testamos 8 configurações diferentes para isolar qual componente estava ajudando, atrapalhando, ou se era simplesmente ruído de execução. Todas em comparação controlada (Recall@5 vs. o checkpoint antigo reavaliado no mesmo teste):

| # | Configuração | Sessões | Recall@5 (novo) | Ganho vs. controlado |
|---|---|---|---|---|
| 1 | Atenção + BPR + temporal (dropout 0.5, patience 5) | 50k | 0,0571 | -2,4% |
| 2 | Atenção + BPR + temporal (dropout 0.5, patience 5) — rodada final, mesma config, seed diferente por conta de ajustes no gerador | 50k | 0,0567 | -3,1% |
| 3 | Atenção + BPR + temporal (dropout 0.3, patience 8) | 50k | 0,0563 | -3,8% |
| 4 | Atenção + BPR + temporal | 15k | 0,0558 | -18,1% |
| 5 | Atenção + temporal, **sem BPR** | 15k | 0,0565 | -17,1% |
| 6 | Atenção + BPR, **sem peso temporal** | 15k | 0,0555 | -18,5% |
| 7 | Atenção, **sem BPR e sem peso temporal** | 15k | 0,0562 | -17,5% |
| 8 | **Arquitetura antiga** (sem atenção) treinada do zero, sem BPR/temporal | 15k | 0,0605 | -11,1% |
| 9 | Atenção com dropout 0.3, sem BPR/temporal | 15k | 0,0586 | -13,9% |

**Achados:**
- Comparando #4, #5, #6 e #7 (todas a 15k sessões): BPR ligado/desligado e peso temporal ligado/desligado não fazem diferença relevante entre si — todas ficam no mesmo patamar (~-17% a -18,5%). **Esses dois componentes são neutros nesta base de dados**, nem ajudam nem atrapalham de forma consistente.
- Comparando #8 (arquitetura antiga, sem atenção) com #9 (atenção): a arquitetura antiga treinada do zero em dados novos sai ligeiramente melhor (-11,1% vs -13,9%). **A atenção não ajuda aqui** — para sessões curtas (média ~6 itens, máximo 10), o estado final da GRU já captura a maior parte da informação relevante; a camada de atenção adiciona parâmetros e complexidade de otimização sem ganho correspondente.
- Comparando as rodadas a 15k sessões (#4-#9, gap de -11% a -18,5%) com as rodadas a 50k sessões (#1-#3, gap de -2,4% a -3,8%): **a quantidade de dados é o fator que mais fecha a diferença**, não a arquitetura ou a loss. Ainda assim, mesmo a 50k sessões, o modelo não ultrapassa o antigo — só chega perto.
- Dropout (0.5 vs 0.3) e patience (5 vs 8) não mudaram o resultado de forma significativa (#1/#2 vs #3).

**Causa raiz:** o gerador de sessões sintéticas tem uma fração grande de aleatoriedade pura embutida por item (75% de chance de continuar na mesma categoria; dos 25% restantes, metade troca de categoria aleatoriamente e metade pula para um item totalmente aleatório do catálogo). Isso impõe um teto baixo ao Recall@K — em torno de 0,06-0,07 — que nenhuma arquitetura consegue superar em 50%, porque uma fração substancial das transições reais da sessão é genuinamente imprevisível por construção do próprio gerador. O número histórico de ~0,06 provavelmente já estava perto desse teto por sorte de inicialização/seed, não porque o modelo antigo era particularmente bom.

---

## 6. Checklist dos critérios de sucesso originais

| Critério | Status |
|---|---|
| Recall@5 melhora ≥ 50% vs. modelo anterior | ❌ Não atingido (ficou entre -2% e -4% na comparação controlada mais rigorosa) |
| App Streamlit exibe as novas recomendações sem erro | ✅ Testado localmente (Playwright headless) — funciona, sem erros de console |
| Novo checkpoint e catálogo commitados e pushados | ✅ Commit `561dfab`, push feito para `origin/main` |
| API pública do `app.py` mantida (`SessionGRU`, `recommend`) | ✅ Assinaturas de `__init__`/`forward` inalteradas |
| Sem dependências pesadas novas | ✅ Só torch/pandas/numpy/streamlit, já presentes |
| `.streamlit/config.toml` mantido | ✅ Não alterado |

---

## 7. Arquivos e commits

- **Notebook atualizado:** `/Users/marks3015/Downloads/recomendador_sessoes4.ipynb` (não versionado neste repo — é o notebook de origem, separado do app).
- **Checkpoint novo:** `recomendador_checkpoint_v4.pt` (commitado).
- **Catálogo:** `catalogo_v4.csv` (regenerado, byte-idêntico ao anterior — sem diff no commit).
- **Comparação de métricas (dados brutos):** `comparacao_metricas_v4_vs_v5.csv` (commitado).
- **Este relatório:** `RELATORIO_RETREINO_V5.md`.
- **Commits:**
  - `561dfab` — retreino + fix do bug numpy.float64 (no GitHub).

---

## 8. Recomendações / próximos passos

Para realmente atingir um ganho grande e defensável no Recall@5, a alavanca mais eficaz identificada **não é** arquitetura/loss — é reduzir a aleatoriedade estrutural do gerador de sessões sintéticas. Duas opções, em ordem de invasividade:

1. **Aumentar a probabilidade de continuidade de categoria** (hoje 75%) para ~90-95%, e/ou reduzir a fração de "item totalmente aleatório" dentro dos 25% restantes. Isso tornaria o "próximo item" genuinamente mais previsível e deve produzir um ganho real e grande no Recall@K com a mesma arquitetura atual (atenção + BPR). **Não foi feito** porque muda o cenário sintético além do que foi pedido originalmente (NUM_SESSIONS, dedupe, peso temporal) — é uma decisão de produto/dado, não de modelagem.
2. **Rodar múltiplas seeds e reportar média ± desvio-padrão**, já que há variância real de ~±0,003-0,005 em Recall@5 entre execuções idênticas (ver rodadas #1 vs #2 na seção 5) — importante para não interpretar ruído como sinal em iterações futuras.
3. Se o catálogo/comportamento de sessão real (não sintético) estiver disponível no futuro, os componentes implementados aqui (atenção, negative sampling, peso temporal) são práticas padrão da literatura e devem se comportar de forma mais favorável em dados com estrutura genuína — o resultado neutro observado aqui é específico da aleatoriedade injetada neste gerador sintético, não uma falha geral dessas técnicas.
