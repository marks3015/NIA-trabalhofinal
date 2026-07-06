# Design System — Recomendador de Sessões

## Visão Geral

Interface redesenhada seguindo o **Material Design 3 (MD3)** do Google, com foco em usabilidade, acessibilidade e aparência profissional de e-commerce.

## Sistema de Cores

Paleta MD3 Light gerada a partir do **Theme Builder** (`#6750A4` como primary):

| Token | Cor | Uso |
|---|---|---|
| Primary | `#6750A4` | Botões, links, badge, preço |
| On Primary | `#FFFFFF` | Texto sobre primary |
| Primary Container | `#EADDFF` | Hero, seção de recomendações |
| Secondary | `#625B71` | Elementos secundários |
| Secondary Container | `#E8DEF8` | Chips ativos, background de seção |
| Surface | `#FFFBFE` | Fundo da página e cards |
| Surface Variant | `#E7E0EC` | Fundo de itens do carrinho, skeleton |
| Error | `#B3261E` | Remover item |
| Outline | `#79747E` | Bordas de inputs e chips |

## Tipografia

Fonte: **Inter** (Google Fonts)

| Estilo | Tamanho | Peso | Uso |
|---|---|---|---|
| Display | 2rem | 700 | Hero title |
| Headline | 1.5rem | 700 | Section titles |
| Title Large | 1.375rem | 600 | App bar brand |
| Title Medium | 0.9375rem | 600 | Product name |
| Body | 0.875rem | 400 | Subtítulos, descrições |
| Label Large | 0.8125rem | 600 | Botões, chips |
| Label Small | 0.6875rem | 600 | Category badge, rank |

## Componentes

### App Bar
- Marca com ícone gradiente e nome
- Badge do carrinho com animação `pop` ao adicionar item

### Hero Banner
- Gradiente primary-container → secondary-container
- Título e subtítulo contextuais (muda quando carrinho tem itens)

### Product Card
- MD3 Elevation Level 1 (resting) → Level 3 (hover)
- Imagem com zoom suave no hover
- Categoria como chip MD3, preço em primary
- Botão outline com hover → filled state
- Grid responsivo: 4 colunas (desktop) → 2 (tablet) → 1 (mobile)

### Recommendation Card
- Mesma estrutura do product card + badge de rank
- Score exibido sutilmente
- Seção com fundo gradiente destacado

### Category Chips
- Botões outline com estilo MD3 chip
- Estado active com secondary-container
- Link via query param para filtro rápido

### Cart Sidebar
- Itens com background surface-variant
- Botão remover com hover em error-container
- Total em destaque com fundo primary
- Empty state ilustrado

### Empty States
- Ícone grande com opacidade
- Título e mensagem descritiva

## Layout Responsivo

| Breakpoint | Colunas | Grid |
|---|---|---|
| >1100px | 4 | `repeat(auto-fill, minmax(260px, 1fr))` |
| 700-1100px | 3-4 | `repeat(auto-fill, minmax(220px, 1fr))` |
| 480-700px | 2 | `repeat(2, 1fr)` |
| <480px | 1 | `1fr` |

Container centralizado com `max-width: 1280px`.

## Microinterações

| Elemento | Ação | Efeito |
|---|---|---|
| Card | Hover | Sobe 3px + elevation level 3 |
| Imagem | Hover | Zoom 1.06x (0.4s ease) |
| Badge | Item adicionado | `pop` scale 1.35x (0.35s ease) |
| Botão | Click | Primary fill + white text |
| Input | Focus | Outline primary + box-shadow |
| Toast | Add to cart | `st.toast()` nativo Streamlit |
| Skeleton | Loading | Shimmer animation (1.5s) |

## Acessibilidade

- Textos com tamanho mínimo de 14px (0.875rem)
- Contraste WCAG AA em todas as combinações de cor
- Estados `:focus-visible` em todos os elementos interativos
- `alt` text descritivo em todas as imagens
- Labels semânticas em inputs e botões
- `loading="lazy"` em imagens para performance

## Interatividade via Query Params

Como o Streamlit não permite componentes HTML interativos (buttons, links) dentro de markdown, a interação com os cards é feita via **query params** na URL:

| Parâmetro | Ação |
|---|---|
| `?add_item={id}` | Adiciona produto ao carrinho |
| `?remove_item={id}` | Remove produto do carrinho |
| `?category={nome}` | Filtra por categoria |

O handler `handle_query_params()` processa esses parâmetros e dispara `st.toast()` e `st.rerun()` conforme necessário.

## Estrutura do Arquivo

```
app.py
├── Configurações e imports
├── Design Tokens (constantes de cor)
├── CSS customizado (MD3 completo ~450 linhas)
├── Modelo SessionGRU (INALTERADO)
├── load_model_and_catalog (INALTERADO)
├── recommend (INALTERADO)
├── Componentes de UI
│   ├── build_card_html()
│   ├── render_product_grid()
│   ├── render_skeleton_grid()
│   ├── render_empty_state()
│   ├── render_app_bar()
│   ├── render_hero()
│   ├── render_search_section()
│   ├── render_cart_sidebar()
│   └── render_footer()
├── handle_query_params()
└── main()
```
