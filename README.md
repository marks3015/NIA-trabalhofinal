# 🛍️ Recomendador de Sessões — Loja Mockup

Aplicação Streamlit que demonstra um recomendador de próximo item baseado em sessões, treinado com PyTorch e GRU.

## 📦 Estrutura

```
recomendador-sessoes-app/
├── app.py                              # Aplicação Streamlit
├── requirements.txt                    # Dependências
├── recomendador_checkpoint_v4.pt       # Checkpoint do modelo
├── catalogo_v4.csv                     # Catálogo de produtos
└── README.md                           # Este arquivo
```

## 🚀 Como rodar localmente

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute o app:

```bash
streamlit run app.py
```

3. Acesse no navegador: `http://localhost:8501`

## ☁️ Deploy no Streamlit Cloud

1. Crie um repositório no GitHub com os arquivos desta pasta.
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte seu GitHub.
3. Escolha o repositório e o arquivo `app.py`.
4. Clique em **Deploy**.

> Os arquivos `recomendador_checkpoint_v4.pt` e `catalogo_v4.csv` são pequenos o suficiente para ficarem no repositório. Se o modelo crescer no futuro, considere usar [Git LFS](https://git-lfs.github.com/) ou carregar de uma URL.

## 🧠 Sobre o modelo

- Arquitetura: Embedding + GRU de 2 camadas + features de categoria e preço
- Métricas no teste: Recall@5, MRR@5, NDCG@5, Recall@10, MRR@10, NDCG@10, Coverage@10
- O modelo foi treinado no notebook `recomendador_sessoes4.ipynb`.

## 📝 Licença

Projeto acadêmico. Sinta-se livre para usar e modificar.
