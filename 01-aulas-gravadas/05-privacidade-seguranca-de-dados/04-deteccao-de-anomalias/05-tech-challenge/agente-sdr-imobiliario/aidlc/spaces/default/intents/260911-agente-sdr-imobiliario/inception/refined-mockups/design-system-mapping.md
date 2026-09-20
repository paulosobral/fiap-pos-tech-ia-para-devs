# Design System Mapping — Agente SDR Imobiliário B2B

> Estágio Refined Mockups & UX Design (Inception). Fonte: mockups, refined-mockups-questions (Q4-A: Streamlit default).

---

## Design System: Streamlit Default

**Decisão (Q4-A)**: Padrão limpo/neutro do Streamlit default (como definido em wireframes).

---

## 1. Colors

| Role | Hex | RGB | Streamlit Constant |
|------|-----|-----|---------------------|
| Primary | `#4F8EF7` | 79, 142, 247 | `st.config.get_option("theme.primaryColor")` |
| Background | `#FFFFFF` | 255, 255, 255 | `st.config.get_option("theme.backgroundColor")` |
| Secondary Background | `#F0F2F6` | 240, 242, 246 | `st.config.get_option("theme.secondaryBackgroundColor")` |
| Text | `#262730` | 38, 39, 48 | `st.config.get_option("theme.textColor")` |
| Font | `#262730` | 38, 39, 48 | `st.config.get_option("theme.font")` |
| Success | `#00C851` | 0, 200, 81 | N/A (custom) |
| Warning | `#FFBB33` | 255, 187, 51 | N/A (custom) |
| Error | `#FF4444` | 255, 68, 68 | N/A (custom) |
| Info | `#33B5E5` | 51, 181, 229 | N/A (custom) |

---

## 2. Typography

| Element | Size | Weight | Line Height | Streamlit Component |
|---------|------|--------|-------------|---------------------|
| H1 (Title) | 36px | Bold | 1.2 | `st.title()` |
| H2 (Header) | 28px | Bold | 1.3 | `st.header()` |
| H3 (Subheader) | 22px | Bold | 1.4 | `st.subheader()` |
| Body (Regular) | 16px | Regular | 1.5 | `st.write()`, `st.text()` |
| Small (Caption) | 14px | Regular | 1.4 | `st.caption()` |
| Code | 14px | Monospace | 1.4 | `st.code()` |

**Font Family**: System font stack (San Francisco, Segoe UI, Roboto, Helvetica, Arial)

---

## 3. Spacing

| Token | Value | Usage |
|-------|-------|-------|
| `spacing-xs` | 4px | Between icon and text |
| `spacing-sm` | 8px | Between label and input |
| `spacing-md` | 16px | Between sections |
| `spacing-lg` | 24px | Between cards |
| `spacing-xl` | 32px | Between major sections |

---

## 4. Borders

| Token | Value | Usage |
|-------|-------|-------|
| `border-radius-sm` | 4px | Buttons, inputs |
| `border-radius-md` | 8px | Cards, modals |
| `border-radius-lg` | 12px | Containers |
| `border-width` | 1px | All borders |
| `border-color` | `#E0E0E0` | All borders |

---

## 5. Shadows

| Token | Value | Usage |
|-------|-------|-------|
| `shadow-sm` | `0 1px 2px rgba(0,0,0,0.1)` | Cards, buttons |
| `shadow-md` | `0 2px 4px rgba(0,0,0,0.1)` | Modals, dropdowns |
| `shadow-lg` | `0 4px 8px rgba(0,0,0,0.1)` | Notifications |

---

## 6. Component Mapping

### 6.1 Layout Components

| Design Element | Streamlit Component | Props/Config |
|----------------|---------------------|--------------|
| Page Title | `st.title()` | Text: "W Levitt · Agente SDR" |
| Section Header | `st.header()` | Text: "Visão Geral", "Detalhe do Lead" |
| Subsection | `st.subheader()` | Text: "KPIs", "Esteira Kanban" |
| Divider | `st.divider()` | N/A |
| Columns | `st.columns([2, 1])` | 2:1 ratio for KPIs vs alerts |
| Container | `st.container()` | Border: True, padding: lg |

---

### 6.2 Data Display Components

| Design Element | Streamlit Component | Props/Config |
|----------------|---------------------|--------------|
| KPI Card | `st.metric()` | Label, value, delta |
| Table | `st.dataframe()` | Hide index, use_container_width |
| Timeline | Custom (st.write + icons) | Bullet list with emojis |
| Chart | `st.line_chart()`, `st.bar_chart()` | Data, use_container_width |
| Alert | `st.error()`, `st.warning()`, `st.success()` | Message, icon |
| Info Box | `st.info()` | Message, icon |

---

### 6.3 Form Components

| Design Element | Streamlit Component | Props/Config |
|----------------|---------------------|--------------|
| Text Input | `st.text_input()` | Label, key, max_chars |
| Text Area | `st.text_area()` | Label, key, height |
| Select Dropdown | `st.selectbox()` | Label, options, key |
| Multi-Select | `st.multiselect()` | Label, options, key |
| Date Input | `st.date_input()` | Label, key, value |
| Time Input | `st.time_input()` | Label, key, value |
| Radio Button | `st.radio()` | Label, options, key |
| Checkbox | `st.checkbox()` | Label, key |
| Slider | `st.slider()` | Label, min, max, value, key |

---

### 6.4 Action Components

| Design Element | Streamlit Component | Props/Config |
|----------------|---------------------|--------------|
| Primary Button | `st.button()` | Label, type="primary", key |
| Secondary Button | `st.button()` | Label, type="secondary", key |
| Link Button | `st.link_button()` | Label, url, key |
| Download Button | `st.download_button()` | Label, data, file_name |

---

### 6.5 Feedback Components

| Design Element | Streamlit Component | Props/Config |
|----------------|---------------------|--------------|
| Toast | `st.toast()` | Message, icon |
| Progress Bar | `st.progress()` | Value |
| Spinner | `st.spinner()` | Text |
| Status | `st.status()` | State, label |

---

### 6.6 Modal Components

| Design Element | Streamlit Component | Props/Config |
|----------------|---------------------|--------------|
| Dialog | `st.dialog()` | Title, content, key |
| Expander | `st.expander()` | Label, expanded |
| Sidebar | `st.sidebar()` | N/A |

---

## 7. Icon Mapping

| Icon | Emoji | Usage |
|------|-------|-------|
| Refresh | 🔄 | Refresh button |
| Arrow Left | ← | Back button |
| Arrow Right | → | Forward button |
| Warning | ⚠ | Anomaly alert |
| Success | ✅ | Timeline milestone |
| Error | ❌ | Timeline failure |
| External Link | 🔗 | HubSpot integration |
| Close | ✕ | Modal close |
| Info | ℹ️ | Info tooltip |
| Filter | 🔍 | Filter icon (optional) |

---

## 8. Responsive Behavior (Q6-A)

**Desktop Only** (foco em gestores e corretores):

| Breakpoint | Width | Layout |
|------------|-------|--------|
| Desktop | 1024px+ | 2 columns (KPIs left, alerts right) |
| Tablet | 768px-1023px | 1 column (stacked) |
| Mobile | <768px | Not supported |

---

## 9. Accessibility Tokens (Q5-A - WCAG 2.1 AA)

| Token | Value | WCAG 2.1 AA Requirement |
|-------|-------|------------------------|
| `contrast-ratio-text` | 4.5:1 | Minimum for normal text |
| `contrast-ratio-large` | 3:1 | Minimum for large text (18pt+) |
| `focus-ring` | 2px solid #4F8EF7 | Visible focus indicator |
| `touch-target` | 44x44px | Minimum touch target size |
| `skip-link` | Visible on focus | Skip to main content |

---

## 10. Custom Styles (If Needed)

If branding alignment requires custom styling, override Streamlit defaults via:

```python
import streamlit as st

st.set_page_config(
    page_title="W Levitt · Agente SDR",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom theme (optional, if branding requires)
st.config.update(
    theme={
        "primaryColor": "#4F8EF7",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif"
    }
)
```

**Current Decision**: Use Streamlit default (no custom overrides needed).