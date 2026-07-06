import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path("recomendador_checkpoint_v4.pt")
CATALOG_PATH = Path("catalogo_v4.csv")

st.set_page_config(
    page_title="Recomendador de Sessões",
    page_icon="🛍️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Arquitetura do modelo (deve bater com o notebook)
# ---------------------------------------------------------------------------
class SessionGRU(nn.Module):
    def __init__(self, num_items, num_categories, embed_dim, cat_embed_dim, hidden_dim,
                 num_layers, dropout, pad_idx, cat_pad_idx):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_idx)
        self.cat_embedding = nn.Embedding(num_categories + 1, cat_embed_dim, padding_idx=cat_pad_idx)
        self.price_proj = nn.Linear(1, 8)

        input_dim = embed_dim + cat_embed_dim + 8
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, x, lengths, cat_features, price_features):
        item_emb = self.item_embedding(x)
        cat_emb = self.cat_embedding(cat_features)
        price_emb = self.price_proj(price_features.unsqueeze(-1))
        embedded = torch.cat([item_emb, cat_emb, price_emb], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        hidden = hidden[-1]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


# ---------------------------------------------------------------------------
# Cache de carregamento (evita recarregar a cada interação)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_catalog():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    catalogo = pd.read_csv(CATALOG_PATH)

    model = SessionGRU(
        checkpoint["num_items"],
        len(checkpoint["cat_to_idx"]),
        checkpoint["embed_dim"],
        checkpoint["cat_embed_dim"],
        checkpoint["hidden_dim"],
        checkpoint["num_layers"],
        checkpoint["dropout"],
        checkpoint["pad_idx"],
        checkpoint["cat_pad_idx"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    item_to_idx = checkpoint["item_to_idx"]
    idx_to_item = checkpoint["idx_to_item"]

    # Recria tensores de features dos itens
    item_cat = torch.full(
        (checkpoint["num_items"] + 1,),
        fill_value=checkpoint["cat_pad_idx"],
        dtype=torch.long,
    )
    item_price = torch.zeros(checkpoint["num_items"] + 1)

    for _, row in catalogo.iterrows():
        item_id = row["item_id"]
        if item_id in item_to_idx:
            idx = item_to_idx[item_id]
            item_cat[idx] = checkpoint["cat_to_idx"][row["categoria"]]
            item_price[idx] = (row["preco"] - checkpoint["price_mean"]) / checkpoint["price_std"]

    return model, catalogo, item_to_idx, idx_to_item, item_cat, item_price


# ---------------------------------------------------------------------------
# Função de recomendação
# ---------------------------------------------------------------------------
def recommend(session_items, model, catalogo, item_to_idx, idx_to_item, item_cat, item_price, k=5):
    known = [i for i in session_items if i in item_to_idx]
    if not known:
        return pd.DataFrame()

    indexed = [item_to_idx[item] for item in known]
    x = torch.tensor([indexed], dtype=torch.long)
    lengths = torch.tensor([len(indexed)], dtype=torch.long)
    cat_x = item_cat[x]
    price_x = item_price[x]

    with torch.no_grad():
        logits = model(x, lengths, cat_x, price_x)
        probs = torch.softmax(logits, dim=1)

    # Remove itens já vistos
    seen = set(indexed)
    num_items = len(idx_to_item)
    for idx in seen:
        probs[0, idx] = -float("inf")

    top_k = min(k, num_items - len(seen))
    top_probs, top_idx = torch.topk(probs, top_k, dim=1)

    recommended_ids = [idx_to_item[idx] for idx in top_idx[0].tolist()]
    scores = top_probs[0].tolist()

    resultados = catalogo[catalogo["item_id"].isin(recommended_ids)].copy()
    resultados["score"] = resultados["item_id"].map(dict(zip(recommended_ids, scores)))
    resultados = resultados.sort_values("score", ascending=False).reset_index(drop=True)
    resultados["rank"] = range(1, len(resultados) + 1)

    return resultados[["rank", "item_id", "nome", "categoria", "preco", "score"]]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.title("🛍️ Loja Mockup — Recomendador de Sessões")
    st.markdown(
        "Simule a navegação de um cliente e veja as recomendações geradas pelo modelo GRU treinado."
    )

    model, catalogo, item_to_idx, idx_to_item, item_cat, item_price = load_model_and_catalog()

    # Estado da sessão
    if "session" not in st.session_state:
        st.session_state.session = []

    col_main, col_sidebar = st.columns([3, 1])

    with col_sidebar:
        st.header("📌 Sua sessão")
        if not st.session_state.session:
            st.info("Clique em produtos para montar seu histórico.")
        else:
            for item_id in st.session_state.session:
                prod = catalogo[catalogo["item_id"] == item_id].iloc[0]
                st.write(f"• {prod['nome']}")

        if st.button("🗑️ Limpar sessão", use_container_width=True):
            st.session_state.session = []
            st.rerun()

        st.divider()
        st.caption(
            f"Modelo: GRU com {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parâmetros"
        )

    with col_main:
        # Escolha inicial ou adição de produto
        st.subheader("1️⃣ Escolha um produto para começar")
        opcoes = catalogo.sort_values("nome")
        opcoes_display = [f"{row['nome']} ({row['categoria']}) - R$ {row['preco']:.2f}" for _, row in opcoes.iterrows()]
        opcoes_ids = opcoes["item_id"].tolist()

        escolhido_display = st.selectbox("Produto", opcoes_display, key="produto_select")
        escolhido_id = opcoes_ids[opcoes_display.index(escolhido_display)]

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("➕ Adicionar à sessão", use_container_width=True):
                st.session_state.session.append(escolhido_id)
                st.rerun()

        # Recomendações
        if st.session_state.session:
            st.subheader("2️⃣ Quem viu isso também viu")
            recs = recommend(
                st.session_state.session,
                model,
                catalogo,
                item_to_idx,
                idx_to_item,
                item_cat,
                item_price,
                k=5,
            )

            if recs.empty:
                st.warning("Não foi possível gerar recomendações para essa sessão.")
            else:
                cols = st.columns(len(recs))
                for idx, (_, row) in enumerate(recs.iterrows()):
                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div style="border:1px solid #ddd; border-radius:10px; padding:15px; text-align:center;">
                                <h4>#{int(row['rank'])}</h4>
                                <p><strong>{row['nome']}</strong></p>
                                <p style="color:#666;">{row['categoria']}</p>
                                <p style="font-size:1.2em; font-weight:bold;">R$ {row['preco']:.2f}</p>
                                <p style="font-size:0.9em; color:#999;">score: {row['score']:.4f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button("Ver produto", key=f"rec_{row['item_id']}"):
                            st.session_state.session.append(int(row["item_id"]))
                            st.rerun()

                with st.expander("📊 Ver tabela de recomendações"):
                    st.dataframe(recs, use_container_width=True)


if __name__ == "__main__":
    main()
