import random
from pathlib import Path
from urllib.parse import quote, unquote
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
CHECKPOINT_PATH = Path('recomendador_checkpoint_v4.pt')
CATALOG_PATH = Path('catalogo_v4.csv')
st.set_page_config(page_title='Loja Mockup | Recomendador de Sessões', page_icon='🛍️', layout='wide', initial_sidebar_state='expanded')
PRIMARY = '#6750A4'
ON_PRIMARY = '#FFFFFF'
PRIMARY_CONTAINER = '#EADDFF'
ON_PRIMARY_CONTAINER = '#21005D'
SECONDARY = '#625B71'
ON_SECONDARY = '#FFFFFF'
SECONDARY_CONTAINER = '#E8DEF8'
ON_SECONDARY_CONTAINER = '#1D192B'
ERROR = '#B3261E'
ERROR_CONTAINER = '#F9DEDC'
BACKGROUND = '#FFFBFE'
ON_BACKGROUND = '#1C1B1F'
SURFACE = '#FFFBFE'
ON_SURFACE = '#1C1B1F'
SURFACE_VARIANT = '#E7E0EC'
ON_SURFACE_VARIANT = '#49454F'
OUTLINE = '#79747E'
OUTLINE_VARIANT = '#CAC4D0'
SHADOW = '#000000'
CUSTOM_CSS = f"""\n    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');\n\n    :root {{\n        --md-primary: {PRIMARY};\n        --md-on-primary: {ON_PRIMARY};\n        --md-primary-container: {PRIMARY_CONTAINER};\n        --md-on-primary-container: {ON_PRIMARY_CONTAINER};\n        --md-secondary: {SECONDARY};\n        --md-on-secondary: {ON_SECONDARY};\n        --md-secondary-container: {SECONDARY_CONTAINER};\n        --md-on-secondary-container: {ON_SECONDARY_CONTAINER};\n        --md-error: {ERROR};\n        --md-error-container: {ERROR_CONTAINER};\n        --md-background: {BACKGROUND};\n        --md-on-background: {ON_BACKGROUND};\n        --md-surface: {SURFACE};\n        --md-on-surface: {ON_SURFACE};\n        --md-surface-variant: {SURFACE_VARIANT};\n        --md-on-surface-variant: {ON_SURFACE_VARIANT};\n        --md-outline: {OUTLINE};\n        --md-outline-variant: {OUTLINE_VARIANT};\n        --md-shadow: {SHADOW};\n        --md-shape-xs: 4px;\n        --md-shape-sm: 8px;\n        --md-shape-md: 12px;\n        --md-shape-lg: 16px;\n        --md-shape-xl: 28px;\n        --md-elevation-1: 0 1px 3px 1px rgba(0,0,0,0.15), 0 1px 2px 0 rgba(0,0,0,0.30);\n        --md-elevation-2: 0 2px 6px 2px rgba(0,0,0,0.15), 0 1px 2px 0 rgba(0,0,0,0.30);\n        --md-elevation-3: 0 4px 8px 3px rgba(0,0,0,0.15), 0 1px 3px 0 rgba(0,0,0,0.30);\n        --md-elevation-4: 0 6px 10px 4px rgba(0,0,0,0.15), 0 2px 4px 0 rgba(0,0,0,0.30);\n        --max-width: 1280px;\n    }}\n\n    * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}\n\n    html, body, [class*="css"] {{\n        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;\n        background-color: var(--md-background);\n        color: var(--md-on-background);\n    }}\n\n    .main .block-container {{\n        max-width: var(--max-width);\n        padding: 0 1rem !important;\n    }}\n\n    /* ===== Top App Bar ===== */\n    .app-bar {{\n        display: flex;\n        align-items: center;\n        justify-content: space-between;\n        padding: 0.75rem 0;\n        margin-bottom: 1.5rem;\n        border-bottom: 1px solid var(--md-outline-variant);\n    }}\n\n    .app-bar-brand {{\n        display: flex;\n        align-items: center;\n        gap: 0.75rem;\n        text-decoration: none;\n    }}\n\n    .app-bar-brand-icon {{\n        width: 40px;\n        height: 40px;\n        background: linear-gradient(135deg, var(--md-primary), var(--md-secondary));\n        border-radius: var(--md-shape-md);\n        display: flex;\n        align-items: center;\n        justify-content: center;\n        font-size: 1.25rem;\n        color: white;\n        flex-shrink: 0;\n    }}\n\n    .app-bar-brand-text {{\n        font-size: 1.375rem;\n        font-weight: 600;\n        line-height: 1.4;\n        color: var(--md-on-surface);\n        margin: 0;\n    }}\n\n    .app-bar-brand-sub {{\n        font-size: 0.75rem;\n        color: var(--md-on-surface-variant);\n        margin: 0;\n        font-weight: 400;\n    }}\n\n    .app-bar-actions {{\n        display: flex;\n        align-items: center;\n        gap: 0.5rem;\n    }}\n\n    /* ===== Cart Badge ===== */\n    .cart-badge-btn {{\n        position: relative;\n        display: flex;\n        align-items: center;\n        justify-content: center;\n        width: 40px;\n        height: 40px;\n        border-radius: 50%;\n        background: var(--md-surface-variant);\n        border: none;\n        cursor: default;\n        font-size: 1.25rem;\n        transition: background 0.2s;\n    }}\n\n    .cart-badge-count {{\n        position: absolute;\n        top: -4px;\n        right: -4px;\n        background: var(--md-primary);\n        color: white;\n        font-size: 0.6875rem;\n        font-weight: 600;\n        min-width: 18px;\n        height: 18px;\n        border-radius: 9px;\n        display: flex;\n        align-items: center;\n        justify-content: center;\n        padding: 0 4px;\n        box-shadow: 0 1px 3px rgba(0,0,0,0.3);\n        transition: transform 0.2s ease;\n    }}\n\n    .cart-badge-count.pop {{\n        animation: badgePop 0.35s ease;\n    }}\n\n    @keyframes badgePop {{\n        0% {{ transform: scale(1); }}\n        40% {{ transform: scale(1.35); }}\n        100% {{ transform: scale(1); }}\n    }}\n\n    /* ===== Hero ===== */\n    .hero {{\n        background: linear-gradient(135deg, var(--md-primary-container) 0%, var(--md-secondary-container) 100%);\n        border-radius: var(--md-shape-lg);\n        padding: 2.5rem;\n        margin-bottom: 2rem;\n    }}\n\n    .hero-title {{\n        font-size: 2rem;\n        font-weight: 700;\n        line-height: 1.2;\n        color: var(--md-on-primary-container);\n        margin: 0 0 0.5rem 0;\n    }}\n\n    .hero-subtitle {{\n        font-size: 1rem;\n        line-height: 1.5;\n        color: var(--md-on-surface-variant);\n        margin: 0;\n        max-width: 520px;\n    }}\n\n    /* ===== Section ===== */\n    .section-title {{\n        font-size: 1.5rem;\n        font-weight: 700;\n        line-height: 1.3;\n        color: var(--md-on-surface);\n        margin: 1.5rem 0 0.25rem 0;\n    }}\n\n    .section-subtitle {{\n        font-size: 0.875rem;\n        color: var(--md-on-surface-variant);\n        margin: 0 0 1rem 0;\n    }}\n\n    /* ===== Product Grid (CSS Grid responsivo) ===== */\n    .product-grid, .rec-grid {{\n        display: grid;\n        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));\n        gap: 1rem;\n        margin-bottom: 0.5rem;\n    }}\n\n    @media (max-width: 1100px) {{\n        .product-grid, .rec-grid {{ grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }}\n    }}\n\n    @media (max-width: 700px) {{\n        .product-grid, .rec-grid {{ grid-template-columns: repeat(2, 1fr); gap: 0.75rem; }}\n    }}\n\n    @media (max-width: 480px) {{\n        .product-grid, .rec-grid {{ grid-template-columns: 1fr; gap: 0.75rem; }}\n    }}\n\n    /* ===== Product Card ===== */\n    .product-card {{\n        background: var(--md-surface);\n        border-radius: var(--md-shape-md);\n        box-shadow: var(--md-elevation-1);\n        overflow: hidden;\n        transition: box-shadow 0.25s ease, transform 0.25s ease;\n        display: flex;\n        flex-direction: column;\n    }}\n\n    .product-card:hover {{\n        box-shadow: var(--md-elevation-3);\n        transform: translateY(-3px);\n    }}\n\n    .product-card-image-wrapper {{\n        position: relative;\n        width: 100%;\n        aspect-ratio: 1;\n        overflow: hidden;\n        background: var(--md-surface-variant);\n    }}\n\n    .product-card-image {{\n        width: 100%;\n        height: 100%;\n        object-fit: cover;\n        transition: transform 0.4s ease;\n        display: block;\n    }}\n\n    .product-card:hover .product-card-image {{\n        transform: scale(1.06);\n    }}\n\n    .product-card-body {{\n        padding: 0.875rem;\n        display: flex;\n        flex-direction: column;\n        flex: 1;\n    }}\n\n    .product-card-category {{\n        display: inline-block;\n        background: var(--md-secondary-container);\n        color: var(--md-on-secondary-container);\n        padding: 0.15rem 0.55rem;\n        border-radius: var(--md-shape-xs);\n        font-size: 0.6875rem;\n        font-weight: 600;\n        align-self: flex-start;\n        margin-bottom: 0.4rem;\n    }}\n\n    .product-card-name {{\n        font-size: 0.9375rem;\n        font-weight: 600;\n        line-height: 1.3;\n        color: var(--md-on-surface);\n        margin: 0 0 0.2rem 0;\n        display: -webkit-box;\n        -webkit-line-clamp: 2;\n        -webkit-box-orient: vertical;\n        overflow: hidden;\n    }}\n\n    .product-card-price {{\n        font-size: 1.25rem;\n        font-weight: 700;\n        color: var(--md-primary);\n        margin: auto 0 0.65rem 0;\n    }}\n\n    .product-card-btn {{\n        display: flex;\n        align-items: center;\n        justify-content: center;\n        gap: 0.4rem;\n        width: 100%;\n        padding: 0.55rem 0.875rem;\n        border-radius: var(--md-shape-xl);\n        font-size: 0.8125rem;\n        font-weight: 600;\n        border: 1px solid var(--md-outline);\n        background: transparent;\n        color: var(--md-primary);\n        cursor: pointer;\n        text-decoration: none;\n        transition: all 0.2s ease;\n        box-sizing: border-box;\n    }}\n\n    .product-card-btn:hover {{\n        background: var(--md-primary-container);\n        border-color: var(--md-primary);\n    }}\n\n    .product-card-btn:active {{\n        background: var(--md-primary);\n        color: white;\n        border-color: var(--md-primary);\n    }}\n\n    .product-card-btn.primary {{\n        background: var(--md-primary);\n        border-color: var(--md-primary);\n        color: white;\n    }}\n\n    .product-card-btn.primary:hover {{\n        box-shadow: var(--md-elevation-1);\n    }}\n\n    /* ===== Recommendation Section ===== */\n    .rec-section {{\n        background: linear-gradient(135deg, var(--md-primary-container) 0%, var(--md-secondary-container) 100%);\n        border-radius: var(--md-shape-lg);\n        padding: 1.75rem;\n        margin: 2rem 0;\n    }}\n\n    .rec-section .section-title {{\n        color: var(--md-on-primary-container);\n        margin-top: 0;\n    }}\n\n    .rec-section .section-subtitle {{\n        color: var(--md-on-surface-variant);\n    }}\n\n    /* ===== Recommendation Card ===== */\n    .rec-card {{\n        position: relative;\n        background: var(--md-surface);\n        border-radius: var(--md-shape-md);\n        box-shadow: var(--md-elevation-2);\n        overflow: hidden;\n        transition: box-shadow 0.25s ease, transform 0.25s ease;\n        display: flex;\n        flex-direction: column;\n    }}\n\n    .rec-card:hover {{\n        box-shadow: var(--md-elevation-4);\n        transform: translateY(-3px);\n    }}\n\n    .rec-card .product-card-image-wrapper {{\n        aspect-ratio: 1;\n        overflow: hidden;\n        background: var(--md-surface-variant);\n    }}\n\n    .rec-card .product-card-image {{\n        width: 100%;\n        height: 100%;\n        object-fit: cover;\n        transition: transform 0.4s ease;\n        display: block;\n    }}\n\n    .rec-card:hover .product-card-image {{\n        transform: scale(1.06);\n    }}\n\n    .rec-card-badge {{\n        position: absolute;\n        top: 8px;\n        left: 8px;\n        background: var(--md-primary);\n        color: white;\n        padding: 0.15rem 0.5rem;\n        border-radius: var(--md-shape-xs);\n        font-size: 0.6875rem;\n        font-weight: 600;\n        z-index: 2;\n        line-height: 1.4;\n    }}\n\n    .rec-card .product-card-body {{\n        padding: 0.875rem;\n    }}\n\n    .score-badge {{\n        font-size: 0.6875rem;\n        color: var(--md-on-surface-variant);\n        margin-bottom: 0.5rem;\n    }}\n\n    /* ===== Search + Filters ===== */\n    .search-section {{\n        margin-bottom: 1.5rem;\n    }}\n\n    .search-wrapper {{\n        display: flex;\n        align-items: center;\n        gap: 0.75rem;\n        background: var(--md-surface);\n        border: 1px solid var(--md-outline);\n        border-radius: var(--md-shape-xl);\n        padding: 0 0.75rem;\n        transition: border-color 0.2s, box-shadow 0.2s;\n    }}\n\n    .search-wrapper:focus-within {{\n        border-color: var(--md-primary);\n        box-shadow: 0 0 0 3px rgba(103, 80, 164, 0.18);\n    }}\n\n    .search-icon {{\n        font-size: 1.125rem;\n        opacity: 0.5;\n        flex-shrink: 0;\n    }}\n\n    .search-wrapper input {{\n        border: none !important;\n        background: transparent !important;\n        padding: 0.75rem 0 !important;\n        font-size: 0.9375rem !important;\n        box-shadow: none !important;\n        flex: 1;\n        outline: none;\n    }}\n\n    .search-wrapper input:focus {{\n        box-shadow: none !important;\n        border: none !important;\n    }}\n\n    /* ===== Category Chips ===== */\n    .chips-row {{\n        display: flex;\n        flex-wrap: wrap;\n        gap: 0.5rem;\n        margin-top: 0.75rem;\n    }}\n\n    .chip {{\n        display: inline-flex;\n        align-items: center;\n        gap: 0.3rem;\n        padding: 0.35rem 0.875rem;\n        border-radius: var(--md-shape-xl);\n        font-size: 0.8125rem;\n        font-weight: 500;\n        border: 1px solid var(--md-outline);\n        background: transparent;\n        color: var(--md-on-surface-variant);\n        cursor: pointer;\n        text-decoration: none;\n        transition: all 0.2s ease;\n    }}\n\n    .chip:hover {{\n        background: var(--md-secondary-container);\n        border-color: var(--md-secondary);\n        color: var(--md-on-secondary-container);\n    }}\n\n    .chip.active {{\n        background: var(--md-secondary-container);\n        border-color: var(--md-secondary);\n        color: var(--md-on-secondary-container);\n        font-weight: 600;\n    }}\n\n    /* ===== Results Count ===== */\n    .results-count {{\n        font-size: 0.875rem;\n        color: var(--md-on-surface-variant);\n        margin: 0.5rem 0 0.75rem 0;\n    }}\n\n    /* ===== Cart Sidebar ===== */\n    [data-testid="stSidebar"] {{\n        background: var(--md-surface);\n        border-left: 1px solid var(--md-outline-variant);\n    }}\n\n    [data-testid="stSidebar"] > div:first-child {{\n        padding: 1.5rem 1rem !important;\n    }}\n\n    .cart-title {{\n        font-size: 1.125rem;\n        font-weight: 600;\n        color: var(--md-on-surface);\n        margin: 0 0 0.75rem 0;\n        display: flex;\n        align-items: center;\n        gap: 0.5rem;\n    }}\n\n    .cart-item {{\n        background: var(--md-surface-variant);\n        border-radius: var(--md-shape-sm);\n        padding: 0.7rem 0.75rem;\n        margin-bottom: 0.5rem;\n        position: relative;\n    }}\n\n    .cart-item-name {{\n        font-size: 0.8125rem;\n        font-weight: 600;\n        color: var(--md-on-surface);\n        margin: 0 0 0.1rem 0;\n        padding-right: 1.25rem;\n    }}\n\n    .cart-item-detail {{\n        font-size: 0.75rem;\n        color: var(--md-on-surface-variant);\n        margin: 0;\n    }}\n\n    .cart-item-remove {{\n        position: absolute;\n        top: 0.35rem;\n        right: 0.35rem;\n        background: transparent;\n        border: none;\n        cursor: pointer;\n        color: var(--md-on-surface-variant);\n        font-size: 0.875rem;\n        padding: 0.2rem;\n        border-radius: 50%;\n        line-height: 1;\n        text-decoration: none;\n        transition: all 0.15s;\n    }}\n\n    .cart-item-remove:hover {{\n        background: var(--md-error-container);\n        color: var(--md-error);\n    }}\n\n    .cart-total {{\n        background: var(--md-primary);\n        color: white;\n        border-radius: var(--md-shape-sm);\n        padding: 0.875rem;\n        text-align: center;\n        margin: 0.75rem 0;\n    }}\n\n    .cart-total-label {{\n        font-size: 0.75rem;\n        opacity: 0.85;\n    }}\n\n    .cart-total-value {{\n        font-size: 1.25rem;\n        font-weight: 700;\n    }}\n\n    .cart-empty {{\n        text-align: center;\n        padding: 1.5rem 1rem;\n        color: var(--md-on-surface-variant);\n    }}\n\n    .cart-empty-icon {{\n        font-size: 2.5rem;\n        opacity: 0.4;\n        margin-bottom: 0.5rem;\n    }}\n\n    .cart-empty-text {{\n        font-size: 0.875rem;\n        margin: 0;\n    }}\n\n    .cart-sidebar-divider {{\n        border: none;\n        border-top: 1px solid var(--md-outline-variant);\n        margin: 0.75rem 0;\n    }}\n\n    /* ===== Empty State ===== */\n    .empty-state {{\n        text-align: center;\n        padding: 3rem 1rem;\n    }}\n\n    .empty-state-icon {{\n        font-size: 3.5rem;\n        opacity: 0.35;\n        margin-bottom: 0.75rem;\n    }}\n\n    .empty-state-title {{\n        font-size: 1.25rem;\n        font-weight: 600;\n        color: var(--md-on-surface);\n        margin: 0 0 0.35rem 0;\n    }}\n\n    .empty-state-text {{\n        font-size: 0.875rem;\n        color: var(--md-on-surface-variant);\n        margin: 0;\n    }}\n\n    /* ===== Skeleton ===== */\n    .skeleton-grid {{\n        display: grid;\n        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));\n        gap: 1rem;\n        margin-bottom: 1rem;\n    }}\n\n    .skeleton-card {{\n        background: var(--md-surface);\n        border-radius: var(--md-shape-md);\n        box-shadow: var(--md-elevation-1);\n        overflow: hidden;\n    }}\n\n    .skeleton-image {{\n        width: 100%;\n        aspect-ratio: 1;\n        background: linear-gradient(90deg, var(--md-surface-variant) 25%, #f5f0f8 50%, var(--md-surface-variant) 75%);\n        background-size: 200% 100%;\n        animation: shimmer 1.5s ease-in-out infinite;\n    }}\n\n    .skeleton-body {{\n        padding: 0.875rem;\n    }}\n\n    .skeleton-line {{\n        height: 12px;\n        margin-bottom: 8px;\n        border-radius: var(--md-shape-xs);\n        background: linear-gradient(90deg, var(--md-surface-variant) 25%, #f5f0f8 50%, var(--md-surface-variant) 75%);\n        background-size: 200% 100%;\n        animation: shimmer 1.5s ease-in-out infinite;\n    }}\n\n    .skeleton-line.short {{ width: 50%; }}\n    .skeleton-line.medium {{ width: 75%; }}\n\n    @keyframes shimmer {{\n        0% {{ background-position: 200% 0; }}\n        100% {{ background-position: -200% 0; }}\n    }}\n\n    /* ===== Footer ===== */\n    .footer {{\n        text-align: center;\n        padding: 2rem 0;\n        margin-top: 3rem;\n        border-top: 1px solid var(--md-outline-variant);\n        color: var(--md-on-surface-variant);\n        font-size: 0.8125rem;\n    }}\n\n    .footer strong {{\n        color: var(--md-primary);\n    }}\n\n    /* ===== Helpers ===== */\n    .stButton button {{ font-family: 'Inter', sans-serif !important; }}\n    #MainMenu {{ visibility: hidden; }}\n    footer {{ visibility: hidden; }}\n    header[data-testid="stHeader"] {{ background: transparent; }}\n"""

class SessionGRU(nn.Module):

    def __init__(self, num_items, num_categories, embed_dim, cat_embed_dim, hidden_dim, num_layers, dropout, pad_idx, cat_pad_idx):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_idx)
        self.cat_embedding = nn.Embedding(num_categories + 1, cat_embed_dim, padding_idx=cat_pad_idx)
        self.price_proj = nn.Linear(1, 8)
        input_dim = embed_dim + cat_embed_dim + 8
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, x, lengths, cat_features, price_features):
        item_emb = self.item_embedding(x)
        cat_emb = self.cat_embedding(cat_features)
        price_emb = self.price_proj(price_features.unsqueeze(-1))
        embedded = torch.cat([item_emb, cat_emb, price_emb], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        (_, hidden) = self.gru(packed)
        hidden = hidden[-1]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits

@st.cache_resource
def load_model_and_catalog():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    catalogo = pd.read_csv(CATALOG_PATH)
    model = SessionGRU(checkpoint['num_items'], len(checkpoint['cat_to_idx']), checkpoint['embed_dim'], checkpoint['cat_embed_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'], checkpoint['dropout'], checkpoint['pad_idx'], checkpoint['cat_pad_idx'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    item_to_idx = checkpoint['item_to_idx']
    idx_to_item = checkpoint['idx_to_item']
    item_cat = torch.full((checkpoint['num_items'] + 1,), fill_value=checkpoint['cat_pad_idx'], dtype=torch.long)
    item_price = torch.zeros(checkpoint['num_items'] + 1)
    for (_, row) in catalogo.iterrows():
        item_id = row['item_id']
        if item_id in item_to_idx:
            idx = item_to_idx[item_id]
            item_cat[idx] = checkpoint['cat_to_idx'][row['categoria']]
            item_price[idx] = (row['preco'] - checkpoint['price_mean']) / checkpoint['price_std']
    return (model, catalogo, item_to_idx, idx_to_item, item_cat, item_price)

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
        probs[0, idx] = -float('inf')
    top_k = min(k, num_items - len(seen))
    (top_probs, top_idx) = torch.topk(probs, top_k, dim=1)
    recommended_ids = [idx_to_item[idx] for idx in top_idx[0].tolist()]
    scores = top_probs[0].tolist()
    resultados = catalogo[catalogo['item_id'].isin(recommended_ids)].copy()
    resultados['score'] = resultados['item_id'].map(dict(zip(recommended_ids, scores)))
    resultados = resultados.sort_values('score', ascending=False).reset_index(drop=True)
    resultados['rank'] = range(1, len(resultados) + 1)
    return resultados[['rank', 'item_id', 'nome', 'categoria', 'preco', 'score', 'imagem_url']]

def build_card_html(row, is_recommendation=False):
    """Gera HTML de um card MD3."""
    action_type = 'add_item'
    btn_text = 'Adicionar'
    btn_icon = '🛒'
    btn_class = 'product-card-btn'
    if is_recommendation:
        btn_class += ' primary'
        rank = row.get('rank', '')
        rank_badge = f'<div class="rec-card-badge">#{rank}</div>' if rank else ''
    else:
        rank_badge = ''
    score_html = ''
    if is_recommendation and 'score' in row:
        score_html = f"""<div class="score-badge">Score: {row['score']:.4f}</div>"""
    card_class = 'rec-card' if is_recommendation else 'product-card'
    return f'''\n    <div class="{card_class}">\n        {rank_badge}\n        <div class="product-card-image-wrapper">\n            <img src="{row['imagem_url']}" class="product-card-image" alt="{row['nome']}" loading="lazy">\n        </div>\n        <div class="product-card-body">\n            <span class="product-card-category">{row['categoria']}</span>\n            <h3 class="product-card-name">{row['nome']}</h3>\n            <p class="product-card-price">R$ {row['preco']:.2f}</p>\n            {score_html}\n            <a href="?{action_type}={row['item_id']}" class="{btn_class}">{btn_icon} {btn_text}</a>\n        </div>\n    </div>\n    '''

def render_product_grid(products, is_recommendation=False):
    """Renderiza grid responsivo de cards."""
    if products.empty:
        render_empty_state('Nenhum produto encontrado', 'Tente ajustar sua busca ou explorar outras categorias.')
        return
    cards_html = ''.join((build_card_html(row, is_recommendation) for (_, row) in products.iterrows()))
    grid_class = 'rec-grid' if is_recommendation else 'product-grid'
    st.html(f'<div class="{grid_class}">{cards_html}</div>')

def render_skeleton_grid():
    """Skeleton loading enquanto o modelo carrega."""
    st.html('\n        <div class="skeleton-grid">\n            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>\n            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>\n            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>\n            <div class="skeleton-card"><div class="skeleton-image"></div><div class="skeleton-body"><div class="skeleton-line"></div><div class="skeleton-line short"></div><div class="skeleton-line medium"></div></div></div>\n        </div>\n    ')

def render_empty_state(title, message):
    """Empty state amigável."""
    st.html(f'\n        <div class="empty-state">\n            <div class="empty-state-icon">🔍</div>\n            <div class="empty-state-title">{title}</div>\n            <div class="empty-state-text">{message}</div>\n        </div>\n    ')

def render_app_bar(cart_size, animate_badge=False):
    """App bar com marca e badge do carrinho."""
    badge_class = 'cart-badge-count' + (' pop' if animate_badge else '')
    badge_html = f'<span class="{badge_class}">{cart_size}</span>' if cart_size > 0 else ''
    st.html(f'\n        <div class="app-bar">\n            <div class="app-bar-brand">\n                <div class="app-bar-brand-icon">🛍️</div>\n                <div>\n                    <div class="app-bar-brand-text">Loja Mockup</div>\n                    <div class="app-bar-brand-sub">Recomendador Inteligente</div>\n                </div>\n            </div>\n            <div class="app-bar-actions">\n                <div class="cart-badge-btn">\n                    🛒{badge_html}\n                </div>\n            </div>\n        </div>\n    ')

def render_hero(has_items=False):
    """Hero banner com CTA contextual."""
    if has_items:
        st.html(f'\n            <div class="hero" style="padding:1.5rem 2rem;">\n                <div class="hero-title" style="font-size:1.375rem;">Continue explorando 🚀</div>\n                <div class="hero-subtitle">Adicione mais produtos ao carrinho para recomendações ainda melhores.</div>\n            </div>\n        ')
    else:
        st.html(f'\n            <div class="hero">\n                <div class="hero-title">Seu próximo produto está a um clique 🎯</div>\n                <div class="hero-subtitle">Navegue pelo nosso catálogo, adicione itens ao carrinho e descubra recomendações inteligentes baseadas nas suas escolhas.</div>\n            </div>\n        ')

def render_search_section(search_term, selected_category, categories):
    """Search input + category chips."""
    st.html('<div class="search-section">')
    (col1, col2) = st.columns([3, 1])
    with col1:
        st.html('<div class="search-wrapper"><span class="search-icon">🔍</span>')
        search_term = st.text_input('Buscar', value=search_term, placeholder='Buscar produtos...', label_visibility='collapsed', key='search_input')
        st.html('</div>')
    with col2:
        st.selectbox('Categoria', categories, index=categories.index(selected_category) if selected_category in categories else 0, label_visibility='collapsed', key='cat_select')
    chips_html = '<div class="chips-row">'
    for cat in categories:
        active = cat == st.session_state.get('cat_select', 'Todas')
        chips_html += f"""<a href="?category={quote(cat)}" class="chip{(' active' if active else '')}">{cat}</a>"""
    chips_html += '</div>'
    st.html(chips_html)
    st.html('</div>')
    return search_term

def render_cart_sidebar(catalogo):
    """Sidebar do carrinho com itens, total e ações."""
    st.sidebar.markdown('<div class="cart-title">🛒 Carrinho</div>', unsafe_allow_html=True)
    session = st.session_state.session
    if not session:
        st.sidebar.markdown('\n            <div class="cart-empty">\n                <div class="cart-empty-icon">🛒</div>\n                <p class="cart-empty-text">Seu carrinho está vazio.<br>Adicione produtos para ver recomendações!</p>\n            </div>\n        ', unsafe_allow_html=True)
        return
    total = 0.0
    for item_id in session:
        prod = catalogo[catalogo['item_id'] == item_id]
        if prod.empty:
            continue
        prod = prod.iloc[0]
        total += prod['preco']
        st.sidebar.markdown(f"""\n            <div class="cart-item">\n                <a href="?remove_item={item_id}" class="cart-item-remove" title="Remover">✕</a>\n                <div class="cart-item-name">{prod['nome']}</div>\n                <div class="cart-item-detail">{prod['categoria']} • R$ {prod['preco']:.2f}</div>\n            </div>\n        """, unsafe_allow_html=True)
    st.sidebar.markdown(f'\n        <div class="cart-total">\n            <div class="cart-total-label">Total estimado</div>\n            <div class="cart-total-value">R$ {total:.2f}</div>\n        </div>\n    ', unsafe_allow_html=True)
    st.sidebar.markdown('<hr class="cart-sidebar-divider">', unsafe_allow_html=True)
    if st.sidebar.button('🗑️ Limpar carrinho', use_container_width=True, type='secondary'):
        st.session_state.session = []
        st.rerun()

def render_footer():
    """Footer da aplicação."""
    st.html('\n        <div class="footer">\n            Projeto acadêmico — <strong>Recomendador de Sessões</strong> com PyTorch + GRU\n        </div>\n    ')

def handle_query_params():
    """Processa ações disparadas por links nos cards."""
    params = st.query_params.to_dict()
    if 'add_item' in params:
        try:
            item_id = int(params['add_item'][0])
            prod = st.session_state.catalogo[st.session_state.catalogo['item_id'] == item_id]
            if not prod.empty and item_id not in st.session_state.session:
                st.session_state.session.append(item_id)
                st.toast(f"✅ {prod.iloc[0]['nome']} adicionado ao carrinho!")
            st.query_params.clear()
            st.rerun()
        except (ValueError, IndexError):
            st.query_params.clear()
            st.rerun()
    elif 'remove_item' in params:
        try:
            item_id = int(params['remove_item'][0])
            st.session_state.session = [i for i in st.session_state.session if i != item_id]
            st.query_params.clear()
            st.rerun()
        except ValueError:
            st.query_params.clear()
            st.rerun()

def main():
    with st.spinner('Carregando modelo de recomendação...'):
        (model, catalogo, item_to_idx, idx_to_item, item_cat, item_price) = load_model_and_catalog()
    st.session_state.catalogo = catalogo
    st.session_state.model = model
    if 'session' not in st.session_state:
        st.session_state.session = []
    if 'prev_cart_size' not in st.session_state:
        st.session_state.prev_cart_size = 0
    st.html(f'<style>{CUSTOM_CSS}</style>')
    handle_query_params()
    cart_size = len(st.session_state.session)
    animate_badge = cart_size > st.session_state.prev_cart_size
    st.session_state.prev_cart_size = cart_size
    render_app_bar(cart_size, animate_badge)
    render_hero(has_items=cart_size > 0)
    categories = ['Todas'] + sorted(catalogo['categoria'].unique().tolist())
    if 'search_input' not in st.session_state:
        st.session_state.search_input = ''
    if 'cat_select' not in st.session_state:
        st.session_state.cat_select = 'Todas'
    if 'category' in st.query_params.to_dict():
        cat = st.query_params['category']
        if isinstance(cat, list):
            cat = cat[0]
        cat = unquote(cat)
        if cat in categories:
            st.session_state.cat_select = cat
        st.query_params.clear()
        st.rerun()
    search_term = render_search_section(st.session_state.search_input, st.session_state.cat_select, categories)
    filtered = catalogo.copy()
    if search_term:
        filtered = filtered[filtered['nome'].str.contains(search_term, case=False, na=False)]
    cat_filter = st.session_state.get('cat_select', 'Todas')
    if cat_filter and cat_filter != 'Todas':
        filtered = filtered[filtered['categoria'] == cat_filter]
    st.html(f'<div class="section-title">Produtos</div>')
    if filtered.empty:
        if search_term or (cat_filter and cat_filter != 'Todas'):
            render_empty_state('Nenhum resultado encontrado', f'Não encontramos nada para "{search_term}" na categoria {cat_filter}.' if search_term and cat_filter != 'Todas' else f'Não encontramos nada para "{search_term}".' if search_term else f'Nenhum produto na categoria {cat_filter}.')
        else:
            render_empty_state('Nenhum produto disponível', 'Parece que o catálogo está vazio. Tente novamente mais tarde.')
    else:
        count_msg = f'{len(filtered)} produto(s) encontrado(s)'
        if search_term:
            count_msg += f' para "{search_term}"'
        if cat_filter and cat_filter != 'Todas':
            count_msg += f' em {cat_filter}'
        st.html(f'<div class="results-count">{count_msg}.</div>')
        render_product_grid(filtered)
    if cart_size > 0:
        st.html('<div class="rec-section"><div class="section-title">✨ Quem viu isso também viu</div>')
        session_cats = catalogo[catalogo['item_id'].isin(st.session_state.session)]['categoria'].tolist()
        if session_cats:
            predominant_cat = max(set(session_cats), key=session_cats.count)
            st.html(f'<div class="section-subtitle">Baseado no seu interesse em <strong>{predominant_cat}</strong></div>')
        recs = recommend(st.session_state.session, model, catalogo, item_to_idx, idx_to_item, item_cat, item_price, k=5)
        if recs.empty:
            st.warning('Não foi possível gerar recomendações para essa sessão.')
        else:
            render_product_grid(recs, is_recommendation=True)
            with st.expander('📊 Ver dados técnicos das recomendações'):
                st.dataframe(recs[['rank', 'nome', 'categoria', 'preco', 'score']], use_container_width=True, hide_index=True)
        st.html('</div>')
    render_footer()
    render_cart_sidebar(catalogo)
    st.sidebar.divider()
    st.sidebar.caption(f'⚙️ GRU com {sum((p.numel() for p in model.parameters() if p.requires_grad)):,} parâmetros')
if __name__ == '__main__':
    main()