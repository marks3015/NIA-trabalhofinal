import random
from html import escape
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
GRID_COLUMNS = 4  # cards por linha no desktop (e nº de recomendações na loja)
PAGE_SIZE = 24  # produtos por página no catálogo (6 linhas de 4)
CAROUSEL_RECS = 12  # recomendações no carrossel do checkout

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
PRIMARY_HOVER = "#7B62B4"  # primary + state layer de hover (MD3)
SHIMMER_HIGHLIGHT = "#F5F0F8"  # brilho do shimmer do skeleton

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
        --md-primary-hover: {PRIMARY_HOVER};
        --md-shimmer-highlight: {SHIMMER_HIGHLIGHT};
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
    /* .st-key-app_bar é um st.container(horizontal=True) real: a marca e o
       botão do carrinho são filhos do mesmo flex row. */
    .st-key-app_bar,
    .st-key-app_bar [data-testid="stHorizontalBlock"] {{
        align-items: center;
        justify-content: space-between;
        width: 100%;
    }}

    .st-key-app_bar {{
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
        font-size: 0.8125rem;
        color: var(--md-on-surface-variant);
        margin: 0;
        font-weight: 400;
    }}

    /* ===== Botão do carrinho (app bar) ===== */
    .st-key-cart_button button {{
        width: auto !important;
        min-width: 48px;
        min-height: 40px;
        padding: 0.5rem 1rem !important;
        font-size: 1rem !important;
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

    .hero-compact {{
        padding: 1.5rem 2rem;
    }}

    .hero-compact .hero-title {{
        font-size: 1.375rem;
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

    /* ===== Cards de produto/recomendação =====
       O chrome do card (fundo, sombra, hover) fica no st.container real
       (key="card_product_..." / key="card_rec_...") que envolve o HTML e o
       botão nativo — assim o CTA fica dentro do card, sob a mesma elevação. */
    [class*="st-key-card_"] {{
        position: relative;
        background: var(--md-surface);
        border-radius: var(--md-shape-md);
        box-shadow: var(--md-elevation-1);
        overflow: hidden;
        padding-bottom: 0.875rem;
        gap: 0.5rem;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }}

    [class*="st-key-card_"]:hover {{
        box-shadow: var(--md-elevation-3);
        transform: translateY(-3px);
    }}

    [class*="st-key-card_rec"] {{
        box-shadow: var(--md-elevation-2);
    }}

    [class*="st-key-card_rec"]:hover {{
        box-shadow: var(--md-elevation-4);
    }}

    /* O botão fica dentro do card com margens laterais iguais ao padding do
       corpo. O wrapper ocupa a largura restante (100% - 2×0.875rem) e o
       botão interno ocupa 100% desse wrapper — assim não estoura o card. */
    [class*="st-key-card_"] .stButton {{
        margin: 0 0.875rem;
        width: calc(100% - 1.75rem) !important;
        display: block;
    }}

    [class*="st-key-card_"] .stButton > button {{
        width: 100% !important;
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

    [class*="st-key-card_"]:hover .product-card-image {{
        transform: scale(1.06);
    }}

    .product-card-body {{
        padding: 0 0.875rem;
        display: flex;
        flex-direction: column;
    }}

    .product-card-category {{
        display: inline-block;
        background: var(--md-secondary-container);
        color: var(--md-on-secondary-container);
        padding: 0.15rem 0.55rem;
        border-radius: var(--md-shape-xs);
        font-size: 0.75rem;
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
        margin: 0.2rem 0 0 0;
    }}

    /* ===== Botões — MD3 mapeado nos 3 kinds nativos do Streamlit =====
       secondary = outlined (padrão), primary = filled (CTA de destaque),
       tertiary = text/icon button (ações discretas, ex.: remover item). */
    [data-testid="stButton"] button {{
        width: 100%;
        padding: 0.55rem 0.875rem !important;
        border-radius: var(--md-shape-xl) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        min-height: 40px;
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
        background: var(--md-primary-hover) !important;
        box-shadow: var(--md-elevation-2);
    }}

    [data-testid="stBaseButton-tertiary"] {{
        border: none !important;
        background: transparent !important;
        color: var(--md-on-surface-variant) !important;
        border-radius: 50% !important;
        width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
    }}

    [data-testid="stBaseButton-tertiary"]:hover {{
        background: var(--md-error-container) !important;
        color: var(--md-error) !important;
    }}

    /* Confirmação destrutiva (limpar carrinho) usa a cor de erro do MD3 */
    .st-key-clear_yes button {{
        background: var(--md-error) !important;
        border-color: var(--md-error) !important;
        color: white !important;
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
    /* .st-key-rec_section (loja) e .st-key-checkout_rec_section (checkout)
       são st.container(key=...) reais que envolvem título, grid e expander —
       o mesmo destaque em degradê nas duas telas. */
    .st-key-rec_section,
    .st-key-checkout_rec_section {{
        background: linear-gradient(135deg, var(--md-primary-container) 0%, var(--md-secondary-container) 100%);
        border-radius: var(--md-shape-lg);
        padding: 1.75rem;
        margin: 2rem 0;
    }}

    .st-key-rec_section .section-title,
    .st-key-checkout_rec_section .section-title {{
        color: var(--md-on-primary-container);
        margin-top: 0;
    }}

    .st-key-rec_section .section-subtitle,
    .st-key-checkout_rec_section .section-subtitle {{
        color: var(--md-on-surface-variant);
    }}

    /* ===== Recommendation Card ===== */
    .rec-card-badge {{
        position: absolute;
        top: 8px;
        left: 8px;
        background: var(--md-primary);
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: var(--md-shape-xs);
        font-size: 0.75rem;
        font-weight: 600;
        z-index: 2;
        line-height: 1.4;
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
        background: transparent !important;
    }}

    /* Remove o fundo cinza/lilás nativo do wrapper do Streamlit text_input */
    .st-key-search_wrapper [data-testid="stTextInput"] > div {{
        background: transparent !important;
    }}

    /* ===== Category Chips ===== */
    .st-key-chips_row {{
        flex-wrap: wrap !important;
        justify-content: flex-start !important;
        margin-top: 0.5rem !important;
        gap: 0.5rem !important;
    }}

    .st-key-chips_row [data-testid="stHorizontalBlock"] {{
        justify-content: flex-start !important;
        gap: 0.5rem !important;
    }}

    .st-key-chips_row [data-testid="stColumn"] {{
        flex: 0 0 auto !important;
        min-width: auto !important;
        width: auto !important;
    }}

    .chip-label {{
        font-size: 0.875rem;
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
        border-right: 1px solid var(--md-outline-variant);
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
        font-size: 0.8125rem;
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
        font-size: 0.875rem;
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
        grid-template-columns: repeat(4, 1fr);
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
        background: linear-gradient(90deg, var(--md-surface-variant) 25%, var(--md-shimmer-highlight) 50%, var(--md-surface-variant) 75%);
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
        background: linear-gradient(90deg, var(--md-surface-variant) 25%, var(--md-shimmer-highlight) 50%, var(--md-surface-variant) 75%);
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

    /* ===== Grid responsivo (4 → 2 → 1) =====
       Sobrescreve as colunas nativas do Streamlit dentro dos grids de cards
       (st.container(key="grid_...")) com breakpoints reais; as linhas de 4
       cards quebram de forma uniforme (2+2 e depois 1 por linha). O skeleton
       usa os mesmos breakpoints para não "saltar" quando o conteúdo chega. */
    [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
        row-gap: 1rem;
    }}

    @media (max-width: 1100px) {{
        [class*="st-key-grid_"] [data-testid="stColumn"] {{
            flex: 1 1 calc(50% - 1rem) !important;
            min-width: calc(50% - 1rem) !important;
        }}
        .skeleton-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    @media (max-width: 700px) {{
        .hero {{ padding: 1.75rem 1.5rem; }}
    }}

    @media (max-width: 480px) {{
        [class*="st-key-grid_"] [data-testid="stColumn"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}
        .skeleton-grid {{ grid-template-columns: 1fr; }}
        .hero-title {{ font-size: 1.5rem; }}
    }}

    /* ===== Carrossel de recomendações (checkout) =====
       st.container(key="carousel_...", horizontal=True) com rolagem lateral
       e scroll-snap: os cards têm largura fixa e não quebram linha — deslize
       (touch/trackpad) ou role para o lado para ver mais. */
    [class*="st-key-carousel_"] {{
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        overflow-x: auto;
        overflow-y: hidden;
        scroll-snap-type: x proximity;
        padding: 0.5rem 0.25rem 1rem 0.25rem;
        scrollbar-width: thin;
        scrollbar-color: var(--md-outline-variant) transparent;
    }}

    /* O Streamlit envolve cada filho do bloco horizontal num stLayoutWrapper
       de largura 100% — sem o override abaixo, cada card ganha um vão de
       ~720px. E o wrapper interno é flex de COLUNA, então um flex-basis no
       card viraria altura (cortando corpo e botão): a largura fixa precisa
       vir de width/min/max, nunca de flex-basis. O display:flex + stretch
       fazem todos os cards da linha terem a mesma altura. */
    [class*="st-key-carousel_"] > [data-testid="stLayoutWrapper"],
    [class*="st-key-carousel_"] > div {{
        flex: 0 0 auto !important;
        width: auto !important;
        display: flex;
        align-items: stretch;
    }}

    [class*="st-key-carousel_"] [class*="st-key-card_"] {{
        width: 240px !important;
        min-width: 240px !important;
        max-width: 240px !important;
        scroll-snap-align: start;
    }}

    /* Empurra o botão para a base do card — em cards de mesma altura, os
       CTAs ficam alinhados mesmo quando o título quebra em 2 linhas. */
    [class*="st-key-carousel_"] [class*="st-key-card_"] > [data-testid="stElementContainer"]:last-child {{
        margin-top: auto;
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
    """GRU + atenção aditiva sobre os estados ocultos da sequência: o
    contexto final é uma combinação ponderada de todos os passos (não
    apenas o último), o que ajuda o modelo a focar nos itens mais
    relevantes da sessão em vez de só no mais recente."""

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
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_context = nn.Linear(hidden_dim, 1, bias=False)
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
        packed_output, _ = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )  # (batch, seq_len, hidden_dim)

        seq_len = outputs.size(1)
        arange = torch.arange(seq_len, device=outputs.device).unsqueeze(0)
        mask = arange < lengths.to(outputs.device).unsqueeze(1)  # (batch, seq_len)

        energy = torch.tanh(self.attn_proj(outputs))
        scores = self.attn_context(energy).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (batch, hidden_dim)
        context = self.dropout(context)
        logits = self.fc(context)
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
def fmt_brl(value):
    """Formata um valor em reais no padrão brasileiro (R$ 1.234,56)."""
    return "R$ " + f"{value:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def build_card_html(row, is_recommendation=False):
    """Gera o conteúdo HTML de um card MD3 (imagem + infos); o chrome do
    card — fundo, sombra, hover — vem do st.container que o envolve em
    render_product_card()."""
    rank_badge = ""
    if is_recommendation:
        rank = row.get("rank", "")
        rank_badge = f'<div class="rec-card-badge">#{rank}</div>' if rank else ""

    nome = escape(str(row["nome"]))
    categoria = escape(str(row["categoria"]))
    # via.placeholder.com (usado no catálogo) saiu do ar; placehold.co serve
    # os mesmos placeholders com a mesma sintaxe de URL.
    imagem_url = escape(str(row["imagem_url"]).replace("via.placeholder.com", "placehold.co"))

    return f"""
    {rank_badge}
    <div class="product-card-image-wrapper">
        <img src="{imagem_url}" class="product-card-image" alt="{nome}" loading="lazy">
    </div>
    <div class="product-card-body">
        <span class="product-card-category">{categoria}</span>
        <h3 class="product-card-name">{nome}</h3>
        <p class="product-card-price">{fmt_brl(row['preco'])}</p>
    </div>
    """


def render_product_card(row, key_prefix, is_recommendation=False):
    """Renderiza um card completo: o st.container(key="card_...") recebe o
    chrome MD3 via CSS e envolve o HTML e o botão nativo, para que o CTA
    faça parte do card (mesma sombra e hover)."""
    item_id = int(row["item_id"])
    card_kind = "rec" if is_recommendation else "product"
    clicked = False

    with st.container(key=f"card_{card_kind}_{key_prefix}"):
        st.html(build_card_html(row, is_recommendation))

        if item_id in st.session_state.session:
            st.button(
                "✓ No carrinho",
                key=f"{key_prefix}_{item_id}",
                width="stretch",
                disabled=True,
                help="Este item já está no seu carrinho (máx. 1 unidade nesta demo)",
            )
        else:
            clicked = st.button(
                "🛒 Adicionar ao carrinho",
                key=f"{key_prefix}_{item_id}",
                width="stretch",
                type="primary" if is_recommendation else "secondary",
                help="Adicionar este item ao carrinho",
            )

    if clicked:
        st.session_state.undo_snapshot = None
        st.session_state.session.append(item_id)
        st.toast(f"✅ {row['nome']} adicionado ao carrinho!")
        st.rerun()


def render_product_grid(products, key_prefix, is_recommendation=False):
    """Renderiza grid de cards em ordem de leitura (linha a linha): cada
    linha é um st.columns(GRID_COLUMNS) e o container key="grid_..." ativa
    os breakpoints responsivos (4 → 2 → 1 colunas) do CSS."""
    if products.empty:
        render_empty_state(
            "Nenhum produto encontrado",
            "Tente ajustar sua busca ou explorar outras categorias."
        )
        return

    with st.container(key=f"grid_{key_prefix}"):
        for start in range(0, len(products), GRID_COLUMNS):
            chunk = products.iloc[start:start + GRID_COLUMNS]
            cols = st.columns(GRID_COLUMNS)
            for offset, (_, row) in enumerate(chunk.iterrows()):
                with cols[offset]:
                    render_product_card(row, f"{key_prefix}_{start + offset}", is_recommendation)


def render_product_carousel(products, key_prefix):
    """Renderiza os cards em um carrossel horizontal: o container
    key="carousel_..." recebe rolagem lateral com scroll-snap via CSS."""
    with st.container(key=f"carousel_{key_prefix}", horizontal=True, gap="small"):
        for idx, (_, row) in enumerate(products.iterrows()):
            render_product_card(row, f"{key_prefix}_{idx}", is_recommendation=True)


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
    """Empty state amigável (title/message podem conter HTML já escapado)."""
    st.html(f"""
        <div class="empty-state">
            <div class="empty-state-icon" aria-hidden="true">🔍</div>
            <h3 class="empty-state-title">{title}</h3>
            <p class="empty-state-text">{message}</p>
        </div>
    """)


def render_app_bar(cart_size):
    """App bar com marca (h1) e botão real do carrinho — em qualquer tela
    (inclusive mobile, onde a sidebar fica recolhida) ele leva ao checkout."""
    with st.container(key="app_bar", horizontal=True, vertical_alignment="center"):
        st.html("""
            <div class="app-bar-brand">
                <div class="app-bar-brand-icon" aria-hidden="true">🛍️</div>
                <div>
                    <h1 class="app-bar-brand-text">Loja Mockup</h1>
                    <p class="app-bar-brand-sub">Recomendador Inteligente</p>
                </div>
            </div>
        """, width="content")

        label = f"🛒 {cart_size}" if cart_size > 0 else "🛒"
        if st.button(label, key="cart_button", help="Abrir o carrinho e finalizar a compra"):
            if cart_size > 0:
                st.session_state.view = "checkout"
            else:
                st.toast("🛒 Seu carrinho ainda está vazio.")
            st.rerun()


def render_hero(has_items=False):
    """Hero banner com CTA contextual."""
    if has_items:
        st.html("""
            <div class="hero hero-compact">
                <h2 class="hero-title">Continue explorando 🚀</h2>
                <p class="hero-subtitle">Adicione mais produtos ao carrinho para recomendações ainda melhores.</p>
            </div>
        """)
    else:
        st.html("""
            <div class="hero">
                <h2 class="hero-title">Seu próximo produto está a um clique 🎯</h2>
                <p class="hero-subtitle">Navegue pelo nosso catálogo, adicione itens ao carrinho e descubra recomendações inteligentes baseadas nas suas escolhas.</p>
            </div>
        """)


def render_search_section(categories):
    """Busca + chips de categoria com botões nativos.

    A busca usa um único ``st.container(key=...)`` para agrupar de fato o
    ícone e o campo de texto no mesmo elemento do DOM — antes eles eram
    renderizados como blocos irmãos (um ``st.markdown`` e um ``st.text_input``
    separados), então o CSS do "pill" de busca nunca chegava a se aplicar.

    O valor do campo vive só em st.session_state.search_input (sem ``value=``),
    evitando o aviso do Streamlit sobre default + Session State.
    """
    with st.container(key="search_wrapper", horizontal=True, vertical_alignment="center", gap="small"):
        st.html('<span class="search-icon" aria-hidden="true">🔍</span>', width="content")
        search_term = st.text_input(
            "Buscar produtos",
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


def render_recommendations(catalogo, model, item_to_idx, idx_to_item, item_cat, item_price, heading,
                           k=GRID_COLUMNS, carousel=False):
    """Bloco de recomendações do modelo — grid de 4 na loja (k=GRID_COLUMNS,
    fecha a linha no desktop) ou carrossel horizontal no checkout."""
    session_cats = catalogo[
        catalogo["item_id"].isin(st.session_state.session)
    ]["categoria"].tolist()

    st.html(f'<h2 class="section-title">{heading}</h2>')

    subtitle = ""
    if session_cats:
        predominant_cat = max(set(session_cats), key=session_cats.count)
        subtitle = f"Baseado no seu interesse em <strong>{escape(predominant_cat)}</strong>"
    if carousel:
        subtitle += " · deslize para ver mais →" if subtitle else "Deslize para ver mais →"
    if subtitle:
        st.html(f'<p class="section-subtitle">{subtitle}</p>')

    recs = recommend(
        st.session_state.session,
        model,
        catalogo,
        item_to_idx,
        idx_to_item,
        item_cat,
        item_price,
        k=k,
    )

    if recs.empty:
        st.warning("Não foi possível gerar recomendações para essa sessão.")
        return

    if carousel:
        render_product_carousel(recs, "rec")
    else:
        render_product_grid(recs, "rec", is_recommendation=True)

    with st.expander("📊 Ver dados técnicos das recomendações"):
        st.dataframe(
            recs[["rank", "nome", "categoria", "preco", "score"]],
            width="stretch",
            hide_index=True,
        )


def _render_undo_button():
    """Botão de desfazer a última remoção/limpeza do carrinho."""
    undo = st.session_state.get("undo_snapshot")
    if not undo:
        return
    if st.sidebar.button(
        f"↩️ Desfazer {undo['label']}",
        key="undo_cart",
        width="stretch",
        type="secondary",
        help="Restaurar os itens removidos do carrinho",
    ):
        st.session_state.session = undo["session"]
        st.session_state.undo_snapshot = None
        st.toast("↩️ Itens restaurados no carrinho.")
        st.rerun()


def render_cart_sidebar(catalogo):
    """Sidebar do carrinho com itens, total e ações."""
    st.sidebar.html('<h2 class="cart-title">🛒 Carrinho</h2>')

    session = st.session_state.session
    if not session:
        st.sidebar.html("""
            <div class="cart-empty">
                <div class="cart-empty-icon" aria-hidden="true">🛒</div>
                <p class="cart-empty-text">Seu carrinho está vazio.<br>Adicione produtos para ver recomendações!</p>
            </div>
        """)
        _render_undo_button()
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
                <div class="cart-item-name">{escape(str(prod['nome']))}</div>
                <div class="cart-item-detail">{escape(str(prod['categoria']))} • {fmt_brl(prod['preco'])}</div>
            </div>
        """)
        if col2.button("🗑️", key=f"remove_{item_id}", type="tertiary", help=f"Remover {prod['nome']} do carrinho"):
            st.session_state.undo_snapshot = {
                "session": list(st.session_state.session),
                "label": "remoção",
            }
            st.session_state.session = [i for i in st.session_state.session if i != item_id]
            st.toast(f"🗑️ {prod['nome']} removido do carrinho.")
            st.rerun()

    st.sidebar.html(f"""
        <div class="cart-total">
            <div class="cart-total-label">Total estimado</div>
            <div class="cart-total-value">{fmt_brl(total)}</div>
        </div>
    """)

    _render_undo_button()

    if st.sidebar.button(
        "✅ Finalizar compra",
        width="stretch",
        type="primary",
        help="Ver o resumo do pedido e confirmar a compra",
    ):
        st.session_state.view = "checkout"
        st.rerun()

    st.sidebar.html('<hr class="cart-sidebar-divider">')

    # Ação destrutiva em duas etapas: o clique pede confirmação explícita.
    if st.session_state.get("confirm_clear"):
        st.sidebar.warning("Remover todos os itens do carrinho?")
        col_yes, col_no = st.sidebar.columns(2)
        if col_yes.button("Sim, limpar", key="clear_yes", width="stretch"):
            st.session_state.undo_snapshot = {
                "session": list(st.session_state.session),
                "label": "limpeza",
            }
            st.session_state.session = []
            st.session_state.confirm_clear = False
            st.toast("🗑️ Carrinho esvaziado.")
            st.rerun()
        if col_no.button("Cancelar", key="clear_no", width="stretch"):
            st.session_state.confirm_clear = False
            st.rerun()
    elif st.sidebar.button(
        "🗑️ Limpar carrinho",
        width="stretch",
        type="secondary",
        help="Remove todos os itens do carrinho (pede confirmação)",
    ):
        st.session_state.confirm_clear = True
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
                    <div class="checkout-item-name">{escape(str(item['nome']))}</div>
                    <div class="checkout-item-detail">{escape(str(item['categoria']))}</div>
                </div>
                <div class="checkout-item-price">{fmt_brl(item['preco'])}</div>
            </div>
        """)


def render_checkout_view(catalogo, model, item_to_idx, idx_to_item, item_cat, item_price):
    """Tela de finalização: resumo do pedido, upsell com o modelo e confirmação."""
    if st.button("← Voltar à loja", type="secondary", help="Continuar comprando sem finalizar"):
        st.session_state.view = "shop"
        st.rerun()

    st.html('<h2 class="section-title" style="margin-top:0.75rem;">Finalizar compra</h2>')
    st.html('<p class="section-subtitle">Confira os itens do seu pedido antes de confirmar.</p>')

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
            <span class="checkout-total-value">{fmt_brl(total)}</span>
        </div>
    """)

    with st.container(key="checkout_rec_section"):
        render_recommendations(
            catalogo, model, item_to_idx, idx_to_item, item_cat, item_price,
            heading="✨ Que tal completar seu pedido?",
            k=CAROUSEL_RECS,
            carousel=True,
        )

    if st.button(
        "✅ Confirmar pedido",
        type="primary",
        width="stretch",
        help="Confirma a compra e finaliza o pedido",
    ):
        st.session_state.last_order = {"items": items, "total": total}
        st.session_state.session = []
        st.session_state.undo_snapshot = None
        st.session_state.confirm_clear = False
        st.session_state.view = "confirmed"
        st.toast("🎉 Pedido confirmado!")
        st.rerun()


def render_confirmation_view():
    """Tela de agradecimento exibida após a confirmação do pedido."""
    st.html("""
        <div class="confirmation-box">
            <div class="confirmation-icon" aria-hidden="true">🎉</div>
            <h2 class="confirmation-title">Pedido confirmado!</h2>
            <p class="confirmation-text">
                Obrigado pela compra. Este é um checkout de demonstração —
                nenhum pagamento foi processado.
            </p>
        </div>
    """)

    order = st.session_state.get("last_order")
    if order and order.get("items"):
        _render_order_items(order["items"])
        st.html(f"""
            <div class="checkout-total-bar">
                <span class="checkout-total-label">Total pago</span>
                <span class="checkout-total-value">{fmt_brl(order['total'])}</span>
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

    # O skeleton acima já comunica o carregamento — sem st.spinner duplicado.
    try:
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
    if "undo_snapshot" not in st.session_state:
        st.session_state.undo_snapshot = None
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    # --- Estado atual ---
    cart_size = len(st.session_state.session)

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
    render_app_bar(cart_size)

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

        search_term = render_search_section(categories)

        # ==================================================================
        # CATÁLOGO / PRODUTOS
        # ==================================================================
        filtered = catalogo.copy()
        if search_term:
            filtered = filtered[
                filtered["nome"].str.contains(search_term, case=False, na=False, regex=False)
            ]
        cat_filter = st.session_state.get("cat_select", "Todas")
        if cat_filter and cat_filter != "Todas":
            filtered = filtered[filtered["categoria"] == cat_filter]

        st.html('<h2 class="section-title">Produtos</h2>')

        # Valores interpolados em HTML sempre escapados (o termo de busca é
        # entrada livre do usuário).
        esc_term = escape(search_term) if search_term else ""
        esc_cat = escape(cat_filter) if cat_filter else ""

        if filtered.empty:
            if search_term or (cat_filter and cat_filter != "Todas"):
                render_empty_state(
                    "Nenhum resultado encontrado",
                    f'Não encontramos nada para "{esc_term}" na categoria {esc_cat}.'
                    if search_term and cat_filter != "Todas"
                    else f'Não encontramos nada para "{esc_term}".' if search_term
                    else f'Nenhum produto na categoria {esc_cat}.'
                )
            else:
                render_empty_state(
                    "Nenhum produto disponível",
                    "Parece que o catálogo está vazio. Tente novamente mais tarde."
                )
        else:
            # Paginação "carregar mais": renderiza só uma janela do catálogo
            # (500 cards de uma vez = 500 botões por rerun). O contador de
            # itens visíveis reseta quando a busca ou a categoria mudam.
            filter_sig = (search_term, cat_filter)
            if st.session_state.get("catalog_filter_sig") != filter_sig:
                st.session_state.catalog_filter_sig = filter_sig
                st.session_state.catalog_visible = PAGE_SIZE

            page = filtered.head(st.session_state.catalog_visible)

            count_msg = f"Mostrando {len(page)} de {len(filtered)} produto(s)"
            if search_term:
                count_msg += f' para "{esc_term}"'
            if cat_filter and cat_filter != "Todas":
                count_msg += f" em {esc_cat}"
            st.html(f'<div class="results-count">{count_msg}.</div>')

            render_product_grid(page, "product")

            remaining = len(filtered) - len(page)
            if remaining > 0:
                _, col_center, _ = st.columns([1, 2, 1])
                if col_center.button(
                    f"⬇️ Carregar mais ({remaining} restantes)",
                    key="load_more",
                    width="stretch",
                    help="Mostrar mais produtos do catálogo",
                ):
                    st.session_state.catalog_visible += PAGE_SIZE
                    st.rerun()

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
