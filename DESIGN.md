# Design System — Recomendador de Sessões

## Visão Geral

Interface seguindo o **Material Design 3 (MD3)** do Google, com foco em usabilidade, acessibilidade e aparência profissional de e-commerce. Toda a interação acontece por widgets nativos do Streamlit (`st.button`, `st.text_input`), com o visual aplicado via CSS sobre containers reais (`st.container(key=...)` → classe `.st-key-*`).

## Sistema de Cores

Paleta MD3 Light gerada a partir do **Theme Builder** (`#6750A4` como primary):

| Token | Cor | Uso |
|---|---|---|
| Primary | `#6750A4` | Botões, badge de rank, preço, total |
| Primary Hover | `#7B62B4` | Hover do botão filled (state layer MD3) |
| On Primary | `#FFFFFF` | Texto sobre primary |
| Primary Container | `#EADDFF` | Hero, seção de recomendações |
| Secondary | `#625B71` | Elementos secundários |
| Secondary Container | `#E8DEF8` | Chips ativos, badge de categoria |
| Surface | `#FFFBFE` | Fundo da página e cards |
| Surface Variant | `#E7E0EC` | Fundo de itens do carrinho, skeleton |
| Shimmer Highlight | `#F5F0F8` | Brilho da animação do skeleton |
| Error | `#B3261E` | Remover item, confirmação de limpeza |
| Outline | `#79747E` | Bordas de inputs e chips |

Todas as cores vivem em constantes Python e são expostas como CSS custom properties (`--md-*`) — nenhuma cor é hardcoded fora dos tokens.

## Tipografia

Fonte: **Inter** (Google Fonts)

| Estilo | Tamanho | Peso | Uso |
|---|---|---|---|
| Display | 2rem | 700 | Hero title |
| Headline | 1.5rem | 700 | Section titles |
| Title Large | 1.375rem | 600 | App bar brand, hero compacto |
| Title Medium | 0.9375rem | 600 | Nome do produto |
| Body | 0.875rem | 400 | Subtítulos, labels de total, chips |
| Body Small | 0.8125rem | 400 | Detalhes de item, footer |
| Label Large (botões) | 0.875rem | 600 | Todos os botões |
| Label Medium | 0.75rem | 600 | Badge de categoria e de rank |

Escala mínima: **12px** apenas em labels/badges (equivalente ao Label Medium do MD3); texto informativo ≥ 13px; corpo ≥ 14px.

### Hierarquia semântica

- `h1` — marca na app bar ("Loja Mockup")
- `h2` — hero title, section titles, título do carrinho, confirmação
- `h3` — nome do produto nos cards, título de empty state

## Componentes

### App Bar
- Marca com ícone gradiente (decorativo, `aria-hidden`) e nome em `h1`
- **Botão real do carrinho** (`st.button`, key `cart_button`) com contagem de itens no label; abre o checkout — funciona também em mobile, onde a sidebar fica recolhida. Com carrinho vazio, mostra um toast.
- Layout via `st.container(horizontal=True)` com `justify-content: space-between`

### Hero Banner
- Gradiente primary-container → secondary-container
- Variante `.hero-compact` (classe, sem estilos inline) quando o carrinho tem itens

### Product Card
- O chrome do card (fundo, sombra, hover) fica num `st.container(key="card_product_*"/"card_rec_*")` que envolve o HTML **e o botão nativo** — o CTA participa do card, sob a mesma elevação
- MD3 Elevation Level 1 (resting) → Level 3 (hover); rec cards usam 2 → 4
- Imagem com zoom suave no hover, `alt` descritivo e `loading="lazy"`
- Categoria como chip MD3, preço em primary formatado em pt-BR (`R$ 1.234,56`)
- Botão desabilitado "✓ No carrinho" previne adição duplicada (máx. 1 unidade nesta demo)
- Todo texto vindo do catálogo passa por `html.escape()`

### Recommendation Card
- Mesma estrutura do product card + badge de rank
- **4 recomendações** (`k = GRID_COLUMNS`) para fechar a linha no desktop
- Score bruto do modelo aparece apenas no expander "dados técnicos", não no card
- Seção com fundo gradiente idêntico na loja (`rec_section`) e no checkout (`checkout_rec_section`)

### Category Chips
- Botões nativos outline; estado ativo em primary filled
- Persistência via `st.session_state.cat_select`

### Cart Sidebar
- Itens com background surface-variant; remoção com botão de 40×40px (🗑️)
- Total em destaque com fundo primary
- **Limpar carrinho pede confirmação** em duas etapas ("Sim, limpar" em cor de erro / "Cancelar")
- **Desfazer**: remoção individual e limpeza guardam um snapshot e exibem "↩️ Desfazer" até a próxima mudança no carrinho
- Empty state ilustrado

### Empty States
- Ícone grande decorativo (`aria-hidden`), título `h3` e mensagem descritiva
- Mensagens citam o termo buscado e a categoria ativa (escapados)

### Checkout e Confirmação
- Resumo de itens + barra de total; upsell com recomendações do modelo
- Tela de confirmação repete o resumo e deixa claro que é uma demo

## Layout Responsivo

Grid real por linhas de `st.columns(4)` dentro de `st.container(key="grid_*")`, com media queries que sobrescrevem as colunas nativas do Streamlit — as linhas de 4 quebram de forma uniforme:

| Breakpoint | Colunas |
|---|---|
| >1100px | 4 |
| 480–1100px | 2 (2+2 por linha) |
| <480px | 1 |

O skeleton usa os mesmos breakpoints para não "saltar" quando o conteúdo chega. Container centralizado com `max-width: 1280px`; hero reduz padding em telas pequenas.

### Paginação do catálogo

O catálogo renderiza **24 produtos por vez** (`PAGE_SIZE`, 6 linhas de 4) com botão "⬇️ Carregar mais (N restantes)" centralizado — em vez de montar os 500 cards (e 500 botões) a cada rerun. O contador mostra "Mostrando X de Y produto(s)" e a janela visível reseta automaticamente quando a busca ou a categoria mudam. A quantidade carregada persiste em `st.session_state.catalog_visible` entre reruns (ex.: ao adicionar um item ao carrinho).

## Microinterações

| Elemento | Ação | Efeito |
|---|---|---|
| Card | Hover | Sobe 3px + elevation superior |
| Imagem | Hover | Zoom 1.06x (0.4s ease) |
| Botão | Click | Primary fill + white text |
| Input | Focus | Outline primary + box-shadow |
| Toast | Add/remove/undo/confirm | `st.toast()` nativo |
| Skeleton | Loading | Shimmer (1.5s), único indicador de carga |

## Acessibilidade

- Hierarquia de headings semântica (`h1` → `h2` → `h3`) navegável por leitores de tela
- Labels/badges ≥ 12px, texto informativo ≥ 13px, contraste AA nas combinações de cor (sem `opacity` reduzindo contraste de texto)
- Alvos de toque ≥ 40px em todos os botões (`min-height: 40px`; botão de remover 40×40px)
- Estados `:focus-visible` em todos os elementos interativos
- `alt` descritivo e `loading="lazy"` em todas as imagens
- Emojis decorativos com `aria-hidden="true"`
- Todo conteúdo dinâmico interpolado em HTML passa por `html.escape()` (catálogo e entrada do usuário)
- Busca tolerante a caracteres especiais (`regex=False`)

## Fluxos

- **Loja → Checkout → Confirmação** via `st.session_state.view`; carrinho esvaziado durante o checkout redireciona para a loja
- Ações destrutivas: limpeza exige confirmação; remoção e limpeza têm desfazer
- Duplo submit prevenido pelo estado "✓ No carrinho" e pelo rerun do Streamlit

## Estrutura do Arquivo

```
app.py
├── Configurações e imports (GRID_COLUMNS, paths)
├── Design Tokens (constantes de cor)
├── CSS customizado (MD3 + grid responsivo)
├── Modelo SessionGRU (INALTERADO)
├── load_model_and_catalog (INALTERADO)
├── recommend (INALTERADO)
├── Componentes de UI
│   ├── fmt_brl()
│   ├── build_card_html()
│   ├── render_product_card()
│   ├── render_product_grid()
│   ├── render_skeleton_grid()
│   ├── render_empty_state()
│   ├── render_app_bar()
│   ├── render_hero()
│   ├── render_search_section()
│   ├── render_recommendations()
│   ├── _render_undo_button()
│   ├── render_cart_sidebar()
│   ├── render_checkout_view()
│   ├── render_confirmation_view()
│   └── render_footer()
└── main()
```
