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
    page_title="Loja Mockup | Recomendador de Sessões",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS customizado
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }

        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin: 1.5rem 0 1rem 0;
            color: #1f2937;
        }

        .product-card {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            border: 1px solid #e5e7eb;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        }

        .product-image {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 0.8rem;
        }

        .category-badge {
            display: inline-block;
            background: #f3f4f6;
            color: #4b5563;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .product-name {
            font-weight: 600;
            font-size: 0.95rem;
            color: #111827;
            margin-bottom: 0.3rem;
            line-height: 1.3;
        }

        .product-price {
            font-size: 1.2rem;
            font-weight: 700;
            color: #059669;
            margin-bottom: 0.3rem;
        }

        .score-badge {
            font-size: 0.8rem;
            color: #6b7280;
        }

        .cart-item {
            background: #f9fafb;
            border-radius: 10px;
            padding: 0.7rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #667eea;
        }

        .cart-item-name {
            font-weight: 600;
            font-size: 0.9rem;
            color: #1f2937;
        }

        .cart-item-cat {
            font-size: 0.75rem;
            color: #6b7280;
        }

        .recommendation-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 2rem;
        }

        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #6b7280;
        }

        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #9ca3af;
            font-size: 0.85rem;
            margin-top: 3rem;
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
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

    return resultados[["rank", "item_id", "nome", "categoria", "preco", "score", "imagem_url"]]


# ---------------------------------------------------------------------------
# Componentes de UI
# ---------------------------------------------------------------------------
def render_product_card(row, key_prefix, is_recommendation=False):
    """Renderiza um card de produto."""
    col = st.container()
    with col:
        st.markdown(
            f"""
            <div class="product-card">
                <div>
                    <img src="{row['imagem_url']}" class="product-image" alt="{row['nome']}">
                    <span class="category-badge">{row['categoria']}</span>
                    <div class="product-name">{row['nome']}</div>
                    <div class="product-price">R$ {row['preco']:.2f}</div>
                    {'<div class="score-badge">🔮 score: ' + f"{row['score']:.4f}" + '</div>' if is_recommendation else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        btn_label = "🛒 Adicionar ao carrinho" if not is_recommendation else "👀 Ver este produto"
        if st.button(btn_label, key=f"{key_prefix}_{row['item_id']}", use_container_width=True):
            st.session_state.session.append(int(row["item_id"]))
            st.rerun()


def render_cart_sidebar(catalogo):
    """Renderiza o carrinho/sessão na sidebar."""
    st.sidebar.markdown("<h3 style='margin-top:0'>🛒 Seu carrinho</h3>", unsafe_allow_html=True)

    if not st.session_state.session:
        st.sidebar.markdown(
            '<div style="padding:1rem; background:#f3f4f6; border-radius:10px; color:#6b7280; text-align:center;">'
            'Seu carrinho está vazio.<br>Adicione produtos para ver recomendações!'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    total = 0.0
    for item_id in st.session_state.session:
        prod = catalogo[catalogo["item_id"] == item_id].iloc[0]
        total += prod["preco"]
        st.sidebar.markdown(
            f"""
            <div class="cart-item">
                <div class="cart-item-name">{prod['nome']}</div>
                <div class="cart-item-cat">{prod['categoria']} • R$ {prod['preco']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(f"""
        <div style="background:#1f2937; color:white; padding:0.8rem; border-radius:10px; text-align:center; margin-top:1rem;">
            <div style="font-size:0.85rem; opacity:0.8;">Total estimado</div>
            <div style="font-size:1.3rem; font-weight:700;">R$ {total:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🗑️ Limpar carrinho", use_container_width=True):
        st.session_state.session = []
        st.rerun()


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------
def main():
    model, catalogo, item_to_idx, idx_to_item, item_cat, item_price = load_model_and_catalog()

    # Estado da sessão
    if "session" not in st.session_state:
        st.session_state.session = []

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🛍️ Loja Mockup</h1>
            <p>Experimente nosso recomendador inteligente baseado em sessões de navegação</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar com carrinho
    render_cart_sidebar(catalogo)
    st.sidebar.divider()
    st.sidebar.caption(
        f"🔧 Modelo: GRU com {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parâmetros"
    )

    # Seleção de produto
    st.markdown('<div class="section-title">1️⃣ Escolha um produto</div>', unsafe_allow_html=True)

    # Grid de produtos em destaque (primeiros 8 do catálogo, aleatoriamente)
    featured = catalogo.sample(n=min(8, len(catalogo)), random_state=42).reset_index(drop=True)
    cols = st.columns(4)
    for idx, (_, row) in enumerate(featured.iterrows()):
        with cols[idx % 4]:
            render_product_card(row, "featured")

    # Ou buscar por nome/categoria
    with st.expander("🔍 Ou busque um produto específico"):
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("Buscar por nome", "", placeholder="Ex: abajur, fone, camiseta...")
        with col2:
            categories = ["Todas"] + sorted(catalogo["categoria"].unique().tolist())
            selected_cat = st.selectbox("Categoria", categories)

        filtered = catalogo.copy()
        if search_term:
            filtered = filtered[filtered["nome"].str.contains(search_term, case=False, na=False)]
        if selected_cat != "Todas":
            filtered = filtered[filtered["categoria"] == selected_cat]

        if filtered.empty:
            st.info("Nenhum produto encontrado.")
        else:
            st.markdown(f"<p>{len(filtered)} produto(s) encontrado(s)</p>", unsafe_allow_html=True)
            cols = st.columns(4)
            for idx, (_, row) in enumerate(filtered.head(12).iterrows()):
                with cols[idx % 4]:
                    render_product_card(row, "search")

    # Recomendações
    if st.session_state.session:
        st.markdown(
            '<div class="recommendation-section">'
            '<div class="section-title">✨ Quem viu isso também viu</div>'
            '</div>',
            unsafe_allow_html=True,
        )

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
            # Mostra a categoria predominante da sessão
            session_cats = catalogo[catalogo["item_id"].isin(st.session_state.session)]["categoria"].tolist()
            predominant_cat = max(set(session_cats), key=session_cats.count)
            st.markdown(
                f"<p style='color:#6b7280; margin-bottom:1rem;'>"
                f"Baseado no seu interesse em <strong>{predominant_cat}</strong>"
                f"</p>",
                unsafe_allow_html=True,
            )

            cols = st.columns(len(recs))
            for idx, (_, row) in enumerate(recs.iterrows()):
                with cols[idx]:
                    render_product_card(row, "rec", is_recommendation=True)

            with st.expander("📊 Ver dados técnicos das recomendações"):
                st.dataframe(
                    recs[["rank", "nome", "categoria", "preco", "score"]],
                    use_container_width=True,
                    hide_index=True,
                )

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>🎓 Projeto acadêmico — Recomendador de Sessões com PyTorch + GRU</p>
            <p>Deploy automático via Streamlit Cloud</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
