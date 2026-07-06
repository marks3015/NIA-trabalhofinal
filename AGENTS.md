# AGENTS.md — Recomendador de Sessões

## Visão geral

Este é um projeto acadêmico de um **recomendador de próximo item baseado em sessões**, utilizando PyTorch + GRU. A aplicação demonstra o modelo funcionando por meio de uma interface Streamlit com visual de e-commerce mockup.

- **Repo GitHub:** https://github.com/marks3015/NIA-trabalhofinal
- **Deploy alvo:** Streamlit Cloud
- **Notebook de origem:** `recomendador_sessoes4.ipynb` (não incluso neste repo; gerado separadamente)

## Estrutura do projeto

```
recomendador-sessoes-app/
├── app.py                              # Aplicação Streamlit
├── requirements.txt                    # Dependências Python
├── README.md                           # Documentação para humanos/usuários
├── AGENTS.md                           # Este arquivo
├── .gitignore                          # Arquivos ignorados pelo git
├── recomendador_checkpoint_v4.pt       # Checkpoint do modelo treinado (~1,1 MB)
└── catalogo_v4.csv                     # Catálogo de 500 produtos mockup
```

## Como executar

### Localmente

```bash
cd /Users/marks3015/recomendador-sessoes-app
pip install -r requirements.txt
streamlit run app.py
```

Acesse `http://localhost:8501`.

### Streamlit Cloud

1. Acesse https://share.streamlit.io
2. Conecte a conta GitHub `marks3015`
3. Selecione o repositório `NIA-trabalhofinal`
4. Defina o arquivo principal como `app.py`
5. Clique em **Deploy**

## Dependências principais

- `streamlit` — interface web
- `torch` — inferência do modelo
- `pandas` — manipulação do catálogo
- `numpy` — operações numéricas

## Detalhes do modelo

- **Arquitetura:** Embedding de item + Embedding de categoria + projeção de preço, seguido por GRU de 2 camadas e camada linear de saída.
- **Entrada:** sequência de `item_id`s de uma sessão de navegação.
- **Saída:** scores sobre todos os itens do catálogo; os top-k são recomendados.
- **Itens vistos são removidos** das recomendações.
- **Checkpoint salvo em:** `recomendador_checkpoint_v4.pt`
- **Catálogo salvo em:** `catalogo_v4.csv`

## Notas importantes

- O checkpoint e o catálogo são pequenos o suficiente para ficarem no repositório. Se o modelo crescer, considere Git LFS ou download de URL no startup.
- O `AGENTS.md` deve ser mantido atualizado se houver mudanças na estrutura, dependências ou deploy.
- Não altere a classe `SessionGRU` em `app.py` sem garantir compatibilidade com o checkpoint salvo.

## Comandos úteis

```bash
# Commit e push de alterações
git add .
git commit -m "mensagem"
git push origin main

# Ver status do repo
git status

# Testar app localmente
streamlit run app.py
```
