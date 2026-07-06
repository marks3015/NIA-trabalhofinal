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
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# Design Tokens — Material Design 3 (Light)
# ---------------------------------------------------------------------------
PRIMARY = "#6750A4"
ON_PRIMARY = "#FFFFFF"
PRIMARY_CONTAINER = "#EADDFF"
ON_PRIMARY_CONTAINER = "#21005D"
SECONDARY = "#625B71"
ON_SECONDARY = "#FFFFFF"
SECONDARY_CONTAINER = "#E8DEF8"
ON_SECONDARY_CONTAINER = "#1D192B"
ERROR = "#B3261E"
ERROR_CONTAINER = "#F9DEDC"
BACKGROUND = "#FFFBFE"
ON_BACKGROUND = "#1C1B1F"
SURFACE = "#FFFBFE"
ON_SURFACE = "#1C1B1F"
SURFACE_VARIANT = "#E7E0EC"
ON_SURFACE_VARIANT = "#49454F"
OUTLINE = "#79747E"
OUTLINE_VARIANT = "#CAC4D0"
SHADOW = "#000000"

# ---------------------------------------------------------------------------
# CSS customizado — MD3 completo
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        --md-primary: {PRIMARY};
        --md-on-primary: {ON_PRIMARY};
        --md-primary-container: {PRIMARY_CONTAINER};
        --md-on-primary-container: {ON_PRIMARY_CONTAINER};
        --md-secondary: {SECONDARY};
        --md-on-secondary: {ON_SECONDARY};
        --md-secondary-container: {SECONDARY_CONTAINER};
        --md-on-secondary-container: {ON_SECONDARY_CONTAINER};
        --md-error: {ERROR};
        --md-error-container: {ERROR_CONTAINER};
        --md-background: {BACKGROUND};
        --md-on-background: {ON_BACKGROUND};
        --md-surface: {SURFACE};
        --md-on-surface: {ON_SURFACE};
        --md-surface-variant: {SURFACE_VARIANT};
        --md-on-surface-variant: {ON_SURFACE_VARIANT};
        --md-outline: {OUTLINE};
        --md-outline-variant: {OUTLINE_VARIANT};
        --md-shadow: {SHADOW};
        --md-shape-xs: 4px;
        --md-shape-sm: 8px;
        --md-shape-md: 12px;
        --md-shape-lg: 16px;
        --md-shape-xl: 28px;
        --md-elevation-1: 0 1px 3px 1px rgba(0,0,0,0.15), 0 1px 2px 0 rgba(0,0,0,0.30);
        --md-elevation-2: 0 2px 6px 2px rgba(0,0,0,0.15), 0 1px 2px 0 rgba(0,0,0,0.30);
        --md-elevation-3: 0 4px 8px 3px rgba(0,0,0,0.15), 0 1px 3px 0 rgba(0,0,0,0.30);
        --md-elevation-4: 0 6px 10px 4px rgba(0,0,0,0.15), 0 2px 4px 0 rgba(0,0,0,0.30);
        --max-width: 1280px;
    }}

    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--md-background);
        color: var(--md-on-background);
    }}

    .main .block-container {{
        max-width: var(--max-width);
        padding: 0 1rem !important;
    }}

    /* ===== Top App Bar ===== */
    .app-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--md-outline-variant);
    }}

    .app-bar-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-decoration: none;
    }}

    .app-bar-brand-icon {{
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--md-primary), var(--md-secondary));
        border-radius: var(--md-shape-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        color: white;
        flex-shrink: 0;
    }}

    .app-bar-brand-text {{
        font-size: 1.375rem;
        font-weight: 600;
        line-height: 1.4;
        color: var(--md-on-surface);
        margin: 0;
    }}

    .app-bar-brand-sub {{
        font-size: 0.75rem;
        color: var(--md-on-surface-variant);
        margin: 0;
        font-weight: 400;
    }}

    .app-bar-actions {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ===== Cart Badge ===== */
    .cart-badge-btn {{
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--md-surface-variant);
        border: none;
        cursor: default;
        font-size: 1.25rem;
        transition: background 0.2s;
    }}

    .cart-badge-count {{
        position: absolute;
        top: -4px;
        right: -4px;
        background: var(--md-primary);
        color: white;
        font-size: 0.6875rem;
        font-weight: 600;
        min-width: 18px;
        height: 18px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }}

    .cart-badge-count.pop {{
        animation: badgePop 0.35s ease;
    }}

    @keyframes badgePop {{
        0% {{ transform: scale(1); }}
        40% {{ transform: scale(1.35); }}
        100% {{ transform: scale(1); }}
    }}

    /* ===== Hero ===== */
    .hero {{
        background: linear-gradient(135deg, var(--md-primary-container) 0%, var(--md-secondary-container) 100%);
        border-radius: var(--md-shape-lg);
        padding: 2.5rem;
        margin-bottom: 2rem;
    }}

    .hero-title {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        color: var(--md-on-primary-container);
        margin: 0 0 0.5rem 0;
    }}

    .hero-subtitle {{
        font-size: 1rem;
        line-height: 1.5;
        color: var(--md-on-surface-variant);
        margin: 0;
        max-width: 520px;
    }}

    /* ===== Section ===== */
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.3;
        color: var(--md-on-surface);
        margin: 1.5rem 0 0.25rem 0;
    }}

    .section-subtitle {{
        font-size: 0.875rem;
        color: var(--md-on-surface-variant);
        margin: 0 0 1rem 0;
    }}

    /* ===== Product Card ===== */
    .product-card {{
        background: var(--md-surface);
        border-radius: var(--md-shape-md);
        box-shadow: var(--md-elevation-1);
        overflow: hidden;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        display: flex;
        flex-direction: column;
        margin-bottom: 0.35rem;
    }}

    .product-card:hover {{
        box-shadow: var(--md-elevation-3);
        transform: translateY(-3px);
    }}

    .product-card-image-wrapper {{
        position: relative;
        width: 100%;
        aspect-ratio: 1;
        overflow: hidden;
        background: var(--md-surface-variant);
    }}

    .product-card-image {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
        display: block;
    }}

    .product-card:hover .product-card-image {{
        transform: scale(1.06);
    }}

    .product-card-body {{
        padding: 0.875rem;
        display: flex;
        flex-direction: column;
        flex: 1;
    }}

    .product-card-category {{
        display: inline-block;
        background: var(--md-secondary-container);
        color: var(--md-on-secondary-container);
        padding: 0.15rem 0.55rem;
        border-radius: var(--md-shape-xs);
        font-size: 0.6875rem;
        font-weight: 600;
        align-self: flex-start;
        margin-bottom: 0.4rem;
    }}

    .product-card-name {{
        font-size: 0.9375rem;
        font-weight: 600;
        line-height: 1.3;
        color: var(--md-on-surface);
        margin: 0 0 0.2rem 0;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}

    .product-card-price {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--md-primary);
        margin: auto 0 0.4rem 0;
    }}

    /* ===== Botões — MD3 mapeado nos 3 kinds nativos do Streamlit =====
       secondary = outlined (padrão), primary = filled (CTA de destaque),
       tertiary = text/icon button (ações discretas, ex.: remover item). */
    [data-testid="stButton"] button {{
        width: 100%;
        padding: 0.55rem 0.875rem !important;
        border-radius: var(--md-shape-xl) !important;
        font-size: 0.8125rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stBaseButton-secondary"] {{
        border: 1px solid var(--md-outline) !important;
        background: transparent !important;
        color: var(--md-primary) !important;
    }}

    [data-testid="stBaseButton-secondary"]:hover {{
        background: var(--md-primary-container) !important;
        border-color: var(--md-primary) !important;
    }}

    [data-testid="stBaseButton-primary"] {{
        background: var(--md-primary) !important;
        border: 1px solid var(--md-primary) !important;
        color: white !important;
        box-shadow: var(--md-elevation-1);
    }}

    [data-testid="stBaseButton-primary"]:hover {{
        background: #7b62b4 !important;
        box-shadow: var(--md-elevation-2);
    }}

    [data-testid="stBaseButton-tertiary"] {{
        border: none !important;
        background: transparent !important;
        color: var(--md-on-surface-variant) !important;
        border-radius: 50% !important;
        width: 32px !important;
        min-width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
    }}

    [data-testid="stBaseButton-tertiary"]:hover {{
        background: var(--md-error-container) !important;
        color: var(--md-error) !important;
    }}

    [data-testid="stButton"] button:focus-visible {{
        outline: 2px solid var(--md-primary) !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
    }}

    [data-testid="stButton"] button:disabled {{
        opacity: 1 !important;
        background: var(--md-surface-variant) !important;
        color: var(--md-on-surface-variant) !important;
        border-color: var(--md-outline-variant) !important;
        cursor: not-allowed !important;
    }}

    /* ===== Recommendation Section ===== */
    /* .st-key-rec_section é o st.container(key="rec_section") real que
       envolve o título, o grid e o expander (ver render em main()). */
    .st-key-rec_section {{
        background: linear-gradient(135deg, var(--md-primary-container) 0%, var(--md-secondary-container) 100%);
        border-radius: var(--md-shape-lg);
        padding: 1.75rem;
        margin: 2rem 0;
    }}

    .st-key-rec_section .section-title {{
        color: var(--md-on-primary-container);
        margin-top: 0;
    }}

    .st-key-rec_section .section-subtitle {{
        color: var(--md-on-surface-variant);
    }}

    /* ===== Recommendation Card ===== */
    .rec-card {{
        position: relative;
        background: var(--md-surface);
        border-radius: var(--md-shape-md);
        box-shadow: var(--md-elevation-2);
        overflow: hidden;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
        display: flex;
        flex-direction: column;
        margin-bottom: 0.35rem;
    }}

    .rec-card:hover {{
        box-shadow: var(--md-elevation-4);
        transform: translateY(-3px);
    }}

    .rec-card-badge {{
        position: absolute;
        top: 8px;
        left: 8px;
        background: var(--md-primary);
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: var(--md-shape-xs);
        font-size: 0.6875rem;
        font-weight: 600;
        z-index: 2;
        line-height: 1.4;
    }}

    .score-badge {{
        font-size: 0.6875rem;
        color: var(--md-on-surface-variant);
        margin-bottom: 0.4rem;
    }}

    /* ===== Search + Filters ===== */
    .search-section {{
        margin-bottom: 1.5rem;
    }}

    /* .st-key-search_wrapper é o container real (st.container(key=...)) que
       envolve o ícone e o text_input — só assim os dois ficam de fato
       aninhados no mesmo elemento e o CSS abaixo passa a ter efeito. */
    .st-key-search_wrapper {{
        background: var(--md-surface);
        border: 1px solid var(--md-outline);
        border-radius: var(--md-shape-xl);
        padding: 0.15rem 1rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}

    .st-key-search_wrapper:focus-within {{
        border-color: var(--md-primary);
        box-shadow: 0 0 0 3px rgba(103, 80, 164, 0.18);
    }}

    .search-icon {{
        font-size: 1.125rem;
        opacity: 0.5;
        flex-shrink: 0;
    }}

    .st-key-search_wrapper input {{
        border: none !important;
        background: transparent !important;
        padding: 0.65rem 0 !important;
        font-size: 0.9375rem !important;
        box-shadow: none !important;
        outline: none;
    }}

    .st-key-search_wrapper [data-testid="stTextInput"] {{
        flex: 1;
    }}

    /* ===== Category Chips ===== */
    .st-key-chips_row {{
        flex-wrap: wrap !important;
        margin-top: 0.5rem !important;
    }}

    .chip-label {{
        font-size: 0.8125rem;
        color: var(--md-on-surface-variant);
        white-space: nowrap;
    }}

    /* ===== Results Count ===== */
    .results-count {{
        font-size: 0.875rem;
        color: var(--md-on-surface-variant);
        margin: 0.5rem 0 0.75rem 0;
    }}

    /* ===== Cart Sidebar ===== */
    [data-testid="stSidebar"] {{
        background: var(--md-surface);
        border-left: 1px solid var(--md-outline-variant);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding: 1.5rem 1rem !important;
    }}

    .cart-title {{
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--md-on-surface);
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .cart-item {{
        background: var(--md-surface-variant);
        border-radius: var(--md-shape-sm);
        padding: 0.7rem 0.75rem;
        margin-bottom: 0.5rem;
    }}

    .cart-item-name {{
        font-size: 0.8125rem;
        font-weight: 600;
        color: var(--md-on-surface);
        margin: 0 0 0.1rem 0;
    }}

    .cart-item-detail {{
        font-size: 0.75rem;
        color: var(--md-on-surface-variant);
        margin: 0;
    }}

    .cart-sidebar-divider {{
        border: none;
        border-top: 1px solid var(--md-outline-variant);
        margin: 0.75rem 0;
    }}

    .cart-total {{
        background: var(--md-primary);
        color: white;
        border-radius: var(--md-shape-sm);
        padding: 0.875rem;
        text-align: center;
        margin: 0.75rem 0;
    }}

    .cart-total-label {{
        font-size: 0.75rem;
        opacity: 0.85;
    }}

    .cart-total-value {{
        font-size: 1.25rem;
        font-weight: 700;
    }}

    .cart-empty {{
        text-align: center;
        padding: 1.5rem 1rem;
        color: var(--md-on-surface-variant);
    }}

    .cart-empty-icon {{
        font-size: 2.5rem;
        opacity: 0.4;
        margin-bottom: 0.5rem;
    }}

    .cart-empty-text {{
        font-size: 0.875rem;
        margin: 0;
    }}

    /* ===== Checkout ===== */
    .checkout-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: var(--md-surface);
        border: 1px solid var(--md-outline-variant);
        border-radius: var(--md-shape-sm);
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }}

    .checkout-item-name {{
        font-size: 0.9375rem;
        font-weight: 600;
        color: var(--md-on-surface);
        margin: 0 0 0.15rem 0;
    }}

    .checkout-item-detail {{
        font-size: 0.8125rem;
        color: var(--md-on-surface-variant);
        margin: 0;
    }}

    .checkout-item-price {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--md-primary);
        white-space: nowrap;
    }}

    .checkout-total-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--md-primary);
        color: white;
        border-radius: var(--md-shape-md);
        padding: 1rem 1.25rem;
        margin: 0.5rem 0 1.5rem 0;
    }}

    .checkout-total-label {{
        font-size: 0.875rem;
        opacity: 0.85;
    }}

    .checkout-total-value {{
        font-size: 1.5rem;
        font-weight: 700;
    }}

    /* ===== Confirmação de pedido ===== */
    .confirmation-box {{
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }}

    .confirmation-icon {{
        font-size: 3.5rem;
        margin-bottom: 0.75rem;
    }}

    .confirmation-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--md-on-surface);
        margin: 0 0 0.5rem 0;
    }}

    .confirmation-text {{
        font-size: 0.9375rem;
        color: var(--md-on-surface-variant);
        margin: 0 0 1.5rem 0;
    }}

    /* ===== Empty State ===== */
    .empty-state {{
        text-align: center;
        padding: 3rem 1rem;
    }}

    .empty-state-icon {{
        font-size: 3.5rem;
        opacity: 0.35;
        margin-bottom: 0.75rem;
    }}

    .empty-state-title {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--md-on-surface);
        margin: 0 0 0.35rem 0;
    }}

    .empty-state-text {{
        font-size: 0.875rem;
        color: var(--md-on-surface-variant);
        margin: 0;
    }}

    /* ===== Skeleton ===== */
    .skeleton-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }}

    .skeleton-card {{
        background: var(--md-surface);
        border-radius: var(--md-shape-md);
        box-shadow: var(--md-elevation-1);
        overflow: hidden;
    }}

    .skeleton-image {{
        width: 100%;
        aspect-ratio: 1;
        background: linear-gradient(90deg, var(--md-surface-variant) 25%, #f5f0f8 50%, var(--md-surface-variant) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s ease-in-out infinite;
    }}

    .skeleton-body {{
        padding: 0.875rem;
    }}

    .skeleton-line {{
        height: 12px;
        margin-bottom: 8px;
        border-radius: var(--md-shape-xs);
        background: linear-gradient(90deg, var(--md-surface-variant) 25%, #f5f0f8 50%, var(--md-surface-variant) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s ease-in-out infinite;
    }}

    .skeleton-line.short {{ width: 50%; }}
    .skeleton-line.medium {{ width: 75%; }}

    @keyframes shimmer {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}

    /* ===== Footer ===== */
    .footer {{
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--md-outline-variant);
        color: var(--md-on-surface-variant);
        font-size: 0.8125rem;
    }}

    .footer strong {{
        color: var(--md-primary);
    }}

    /* ===== Helpers ===== */
    .stButton button {{ font-family: 'Inter', sans-serif !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
"""


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
# Cache de carregamento
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
# Função de recomendação (inalterada)
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
def build_card_html(row, is_recommendation=False):
    """Gera HTML visual de um card MD3 (sem botão interativo)."""
    rank_badge = ""
    if is_recommendation:
        rank = row.get("rank", "")
        rank_badge = f'<div class="rec-card-badge">#{rank}</div>' if rank else ""

    score_html = ""
    if is_recommendation and "score" in row:
        score_html = f'<div class="score-badge">Score: {row["score"]:.4f}</div>'

    card_class = "rec-card" if is_recommendation else "product-card"

    return f"""
    <div class="{card_class}">
        {rank_badge}
        <div class="product-card-image-wrapper">
            <img src="{row['imagem_url']}" class="product-card-image" alt="{row['nome']}" loading="lazy">
        </div>
        <div class="product-card-body">
            <span class="product-card-category">{row['categoria']}</span>
            <h3 class="product-card-name">{row['nome']}</h3>
            <p class="product-card-price">R$ {row['preco']:.2f}</p>
            {score_html}
        </div>
    </div>
    """


def render_product_card(row, key_prefix, is_recommendation=False):
    """Renderiza card visual + botão Streamlit nativo para adicionar."""
    st.html(build_card_html(row, is_recommendation))

    item_id = int(row["item_id"])
    already_in_cart = item_id in st.session_state.session

    if already_in_cart:
        st.button(
            "✓ No carrinho",
            key=f"{key_prefix}_{item_id}",
            width="stretch",
            disabled=True,
            help="Este item já está no seu carrinho",
        )
        return

    clicked = st.button(
        "🛒 Adicionar ao carrinho",
        key=f"{key_prefix}_{item_id}",
        width="stretch",
        type="primary" if is_recommendation else "secondary",
        help="Adicionar este item ao carrinho",
    )
    if clicked:
        st.session_state.session.append(item_id)
        st.toast(f"✅ {row['nome']} adicionado ao carrinho!")
        st.rerun()


def render_product_grid(products, key_prefix, is_recommendation=False):
    """Renderiza grid de cards com botões nativos do Streamlit."""
    if products.empty:
        render_empty_state(
            "Nenhum produto encontrado",
            "Tente ajustar sua busca ou explorar outras categorias."
        )
        return

    cols = st.columns(4)
    for idx, (_, row) in enumerate(products.iterrows()):
        with cols[idx % 4]:
            render_product_card(row, f"{key_prefix}_{idx}", is_recommendation)


def render_skeleton_grid():
    """Skeleton loading enquanto o modelo carrega."""
    st.html("""
        <div class="skeleton-grid">
            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>
            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>
            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>
            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>
        </div>
    """)


def render_empty_state(title, message):
    """Empty state amigável."""
    st.html(f"""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-text">{message}</div>
        </div>
    """)


def render_app_bar(cart_size, animate_badge=False):
    """App bar com marca e badge do carrinho."""
    badge_class = "cart-badge-count" + (" pop" if animate_badge else "")
    badge_html = f'<span class="{badge_class}">{cart_size}</span>' if cart_size > 0 else ""

    st.html(f"""
        <div class="app-bar">
            <div class="app-bar-brand">
                <div class="app-bar-brand-icon">🛍️</div>
                <div>
                    <div class="app-bar-brand-text">Loja Mockup</div>
                    <div class="app-bar-brand-sub">Recomendador Inteligente</div>
                </div>
            </div>
            <div class="app-bar-actions">
                <div class="cart-badge-btn">
                    🛒{badge_html}
                </div>
            </div>
        </div>
    """)


def render_hero(has_items=False):
    """Hero banner com CTA contextual."""
    if has_items:
        st.html(f"""
            <div class="hero" style="padding:1.5rem 2rem;">
                <div class="hero-title" style="font-size:1.375rem;">Continue explorando 🚀</div>
                <div class="hero-subtitle">Adicione mais produtos ao carrinho para recomendações ainda melhores.</div>
            </div>
        """)
    else:
        st.html(f"""
            <div class="hero">
                <div class="hero-title">Seu próximo produto está a um clique 🎯</div>
                <div class="hero-subtitle">Navegue pelo nosso catálogo, adicione itens ao carrinho e descubra recomendações inteligentes baseadas nas suas escolhas.</div>
            </div>
        """)


def render_search_section(search_term, selected_category, categories):
    """Busca + chips de categoria com botões nativos.

    A busca usa um único ``st.container(key=...)`` para agrupar de fato o
    ícone e o campo de texto no mesmo elemento do DOM — antes eles eram
    renderizados como blocos irmãos (um ``st.markdown`` e um ``st.text_input``
    separados), então o CSS do "pill" de busca nunca chegava a se aplicar.
    """
    with st.container(key="search_wrapper", horizontal=True, vertical_alignment="center", gap="small"):
        st.html('<span class="search-icon">🔍</span>')
        search_term = st.text_input(
            "Buscar produtos",
            value=search_term,
            placeholder="Buscar produtos pelo nome...",
            label_visibility="collapsed",
            key="search_input",
            help="Digite parte do nome do produto para filtrar o catálogo",
        )

    with st.container(key="chips_row", horizontal=True, vertical_alignment="center", gap="small"):
        st.html('<span class="chip-label">Categorias:</span>')
        for cat in categories:
            active = cat == st.session_state.get("cat_select", "Todas")
            if st.button(cat, key=f"chip_{cat}", type="primary" if active else "secondary"):
                st.session_state.cat_select = cat
                st.rerun()

    return search_term


def render_recommendations(catalogo, model, item_to_idx, idx_to_item, item_cat, item_price, heading):
    """Bloco de recomendações do modelo — reaproveitado na loja e no checkout."""
    session_cats = catalogo[
        catalogo["item_id"].isin(st.session_state.session)
    ]["categoria"].tolist()

    st.html(f'<div class="section-title">{heading}</div>')
    if session_cats:
        predominant_cat = max(set(session_cats), key=session_cats.count)
        st.html(
            f'<div class="section-subtitle">Baseado no seu interesse em <strong>{predominant_cat}</strong></div>'
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
        return

    render_product_grid(recs, "rec", is_recommendation=True)

    with st.expander("📊 Ver dados técnicos das recomendações"):
        st.dataframe(
            recs[["rank", "nome", "categoria", "preco", "score"]],
            width="stretch",
            hide_index=True,
        )


def render_cart_sidebar(catalogo):
    """Sidebar do carrinho com itens, total e ações."""
    st.sidebar.html('<div class="cart-title">🛒 Carrinho</div>')

    session = st.session_state.session
    if not session:
        st.sidebar.html("""
            <div class="cart-empty">
                <div class="cart-empty-icon">🛒</div>
                <p class="cart-empty-text">Seu carrinho está vazio.<br>Adicione produtos para ver recomendações!</p>
            </div>
        """)
        return

    total = 0.0
    for item_id in session:
        prod = catalogo[catalogo["item_id"] == item_id]
        if prod.empty:
            continue
        prod = prod.iloc[0]
        total += prod["preco"]

        col1, col2 = st.sidebar.columns([5, 1], vertical_alignment="center")
        col1.html(f"""
            <div class="cart-item">
                <div class="cart-item-name">{prod['nome']}</div>
                <div class="cart-item-detail">{prod['categoria']} • R$ {prod['preco']:.2f}</div>
            </div>
        """)
        if col2.button("✕", key=f"remove_{item_id}", type="tertiary", help=f"Remover {prod['nome']} do carrinho"):
            st.session_state.session = [i for i in st.session_state.session if i != item_id]
            st.toast(f"🗑️ {prod['nome']} removido do carrinho.")
            st.rerun()

    st.sidebar.html(f"""
        <div class="cart-total">
            <div class="cart-total-label">Total estimado</div>
            <div class="cart-total-value">R$ {total:.2f}</div>
        </div>
    """)

    if st.sidebar.button(
        "✅ Finalizar compra",
        width="stretch",
        type="primary",
        help="Ver o resumo do pedido e confirmar a compra",
    ):
        st.session_state.view = "checkout"
        st.rerun()

    st.sidebar.html('<hr class="cart-sidebar-divider">')

    if st.sidebar.button(
        "🗑️ Limpar carrinho",
        width="stretch",
        type="secondary",
        help="Remove todos os itens do carrinho e recomeça a sessão",
    ):
        st.session_state.session = []
        st.toast("🗑️ Carrinho esvaziado.")
        st.rerun()


def render_footer():
    """Footer da aplicação."""
    st.html("""
        <div class="footer">
            Projeto acadêmico — <strong>Recomendador de Sessões</strong> com PyTorch + GRU
        </div>
    """)


def _render_order_items(items):
    """Lista de itens do pedido (usada no checkout e na confirmação)."""
    for item in items:
        st.html(f"""
            <div class="checkout-item">
                <div>
                    <div class="checkout-item-name">{item['nome']}</div>
                    <div class="checkout-item-detail">{item['categoria']}</div>
                </div>
                <div class="checkout-item-price">R$ {item['preco']:.2f}</div>
            </div>
        """)


def render_checkout_view(catalogo, model, item_to_idx, idx_to_item, item_cat, item_price):
    """Tela de finalização: resumo do pedido, upsell com o modelo e confirmação."""
    if st.button("← Voltar à loja", type="secondary", help="Continuar comprando sem finalizar"):
        st.session_state.view = "shop"
        st.rerun()

    st.html('<div class="section-title" style="margin-top:0.75rem;">Finalizar compra</div>')
    st.html('<div class="section-subtitle">Confira os itens do seu pedido antes de confirmar.</div>')

    session = st.session_state.session
    if not session:
        render_empty_state(
            "Seu carrinho está vazio",
            "Volte à loja e adicione produtos antes de finalizar a compra.",
        )
        return

    items, total = [], 0.0
    for item_id in session:
        prod = catalogo[catalogo["item_id"] == item_id]
        if prod.empty:
            continue
        prod = prod.iloc[0]
        total += prod["preco"]
        items.append({"nome": prod["nome"], "categoria": prod["categoria"], "preco": float(prod["preco"])})

    _render_order_items(items)
    st.html(f"""
        <div class="checkout-total-bar">
            <span class="checkout-total-label">Total do pedido</span>
            <span class="checkout-total-value">R$ {total:.2f}</span>
        </div>
    """)

    with st.container(key="checkout_rec_section"):
        render_recommendations(
            catalogo, model, item_to_idx, idx_to_item, item_cat, item_price,
            heading="✨ Que tal completar seu pedido?",
        )

    if st.button(
        "✅ Confirmar pedido",
        type="primary",
        width="stretch",
        help="Confirma a compra e finaliza o pedido",
    ):
        st.session_state.last_order = {"items": items, "total": total}
        st.session_state.session = []
        st.session_state.view = "confirmed"
        st.toast("🎉 Pedido confirmado!")
        st.rerun()


def render_confirmation_view():
    """Tela de agradecimento exibida após a confirmação do pedido."""
    st.html("""
        <div class="confirmation-box">
            <div class="confirmation-icon">🎉</div>
            <div class="confirmation-title">Pedido confirmado!</div>
            <div class="confirmation-text">
                Obrigado pela compra. Este é um checkout de demonstração —
                nenhum pagamento foi processado.
            </div>
        </div>
    """)

    order = st.session_state.get("last_order")
    if order and order.get("items"):
        _render_order_items(order["items"])
        st.html(f"""
            <div class="checkout-total-bar">
                <span class="checkout-total-label">Total pago</span>
                <span class="checkout-total-value">R$ {order['total']:.2f}</span>
            </div>
        """)

    if st.button("🛍️ Continuar comprando", type="primary", width="stretch"):
        st.session_state.view = "shop"
        st.rerun()


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------
def main():
    # --- CSS (precisa vir antes de qualquer render, inclusive o skeleton) ---
    st.html(f"<style>{CUSTOM_CSS}</style>")

    # --- Loading state ---
    # O skeleton só aparece no primeiro carregamento real (cache miss) da
    # sessão; em reruns subsequentes @st.cache_resource já devolve na hora.
    loading_placeholder = st.empty()
    if "assets_loaded" not in st.session_state:
        with loading_placeholder.container():
            st.html(
                '<div class="section-subtitle">Carregando modelo de recomendação e catálogo...</div>'
            )
            render_skeleton_grid()

    try:
        with st.spinner("Carregando modelo de recomendação..."):
            model, catalogo, item_to_idx, idx_to_item, item_cat, item_price = load_model_and_catalog()
    except FileNotFoundError:
        loading_placeholder.empty()
        st.error(
            "⚠️ Não foi possível carregar o modelo ou o catálogo. "
            f"Verifique se os arquivos `{CHECKPOINT_PATH}` e `{CATALOG_PATH}` "
            "estão presentes no diretório do app."
        )
        st.stop()
    except Exception as exc:
        loading_placeholder.empty()
        st.error(f"⚠️ Ocorreu um erro inesperado ao carregar o app: {exc}")
        st.stop()

    loading_placeholder.empty()
    st.session_state.assets_loaded = True

    # Armazena catálogo para acesso global
    st.session_state.catalogo = catalogo
    st.session_state.model = model

    # --- Estado da sessão ---
    if "session" not in st.session_state:
        st.session_state.session = []
    if "prev_cart_size" not in st.session_state:
        st.session_state.prev_cart_size = 0

    # --- Estado atual ---
    cart_size = len(st.session_state.session)
    animate_badge = cart_size > st.session_state.prev_cart_size
    st.session_state.prev_cart_size = cart_size

    # --- Navegação entre telas (loja / checkout / confirmação) ---
    if "view" not in st.session_state:
        st.session_state.view = "shop"
    # Se o carrinho esvaziar enquanto o usuário está no checkout (ex.: item
    # removido pela sidebar), volta para a loja em vez de mostrar um resumo
    # de pedido vazio.
    if st.session_state.view == "checkout" and cart_size == 0:
        st.session_state.view = "shop"

    # ======================================================================
    # TOP APP BAR
    # ======================================================================
    render_app_bar(cart_size, animate_badge)

    if st.session_state.view == "confirmed":
        # ==================================================================
        # CONFIRMAÇÃO DO PEDIDO
        # ==================================================================
        render_confirmation_view()
    elif st.session_state.view == "checkout":
        # ==================================================================
        # FINALIZAÇÃO DA COMPRA
        # ==================================================================
        render_checkout_view(catalogo, model, item_to_idx, idx_to_item, item_cat, item_price)
    else:
        # ==================================================================
        # HERO
        # ==================================================================
        render_hero(has_items=cart_size > 0)

        # ==================================================================
        # SEARCH + FILTROS
        # ==================================================================
        categories = ["Todas"] + sorted(catalogo["categoria"].unique().tolist())

        # Inicializa filtros se não existirem
        if "search_input" not in st.session_state:
            st.session_state.search_input = ""
        if "cat_select" not in st.session_state:
            st.session_state.cat_select = "Todas"

        search_term = render_search_section(
            st.session_state.search_input,
            st.session_state.cat_select,
            categories,
        )

        # ==================================================================
        # CATÁLOGO / PRODUTOS
        # ==================================================================
        filtered = catalogo.copy()
        if search_term:
            filtered = filtered[
                filtered["nome"].str.contains(search_term, case=False, na=False)
            ]
        cat_filter = st.session_state.get("cat_select", "Todas")
        if cat_filter and cat_filter != "Todas":
            filtered = filtered[filtered["categoria"] == cat_filter]

        st.html('<div class="section-title">Produtos</div>')

        if filtered.empty:
            if search_term or (cat_filter and cat_filter != "Todas"):
                render_empty_state(
                    "Nenhum resultado encontrado",
                    f'Não encontramos nada para "{search_term}" na categoria {cat_filter}.'
                    if search_term and cat_filter != "Todas"
                    else f'Não encontramos nada para "{search_term}".' if search_term
                    else f'Nenhum produto na categoria {cat_filter}.'
                )
            else:
                render_empty_state(
                    "Nenhum produto disponível",
                    "Parece que o catálogo está vazio. Tente novamente mais tarde."
                )
        else:
            count_msg = f"{len(filtered)} produto(s) encontrado(s)"
            if search_term:
                count_msg += f' para "{search_term}"'
            if cat_filter and cat_filter != "Todas":
                count_msg += f" em {cat_filter}"
            st.html(f'<div class="results-count">{count_msg}.</div>')
            render_product_grid(filtered, "product")

        # ==================================================================
        # RECOMENDAÇÕES
        # ==================================================================
        if cart_size > 0:
            # st.container(key=...) garante que o título, o grid e o
            # expander fiquem de fato dentro do mesmo elemento estilizado
            # como .st-key-rec_section — abrir/fechar a div em chamadas
            # separadas deixaria o conteúdo como irmão da div, não como
            # filho, e o fundo em degradê não envolveria nada.
            with st.container(key="rec_section"):
                render_recommendations(
                    catalogo, model, item_to_idx, idx_to_item, item_cat, item_price,
                    heading="✨ Quem viu isso também viu",
                )

    # ======================================================================
    # FOOTER
    # ======================================================================
    render_footer()

    # ======================================================================
    # SIDEBAR — Carrinho
    # ======================================================================
    render_cart_sidebar(catalogo)
    st.sidebar.divider()
    st.sidebar.caption(
        f"⚙️ GRU com "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parâmetros"
    )


if __name__ == "__main__":
    main()
