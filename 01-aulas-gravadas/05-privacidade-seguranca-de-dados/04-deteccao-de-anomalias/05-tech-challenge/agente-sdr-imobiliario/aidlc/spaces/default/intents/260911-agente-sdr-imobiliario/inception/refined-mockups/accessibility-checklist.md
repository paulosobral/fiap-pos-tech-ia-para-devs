# Accessibility Checklist — Agente SDR Imobiliário B2B

> Estágio Refined Mockups & UX Design (Inception). Fonte: mockups, refined-mockups-questions (Q5-A: WCAG 2.1 AA, Q6-A: desktop only).

---

## WCAG 2.1 AA Compliance Checklist

**Level**: WCAG 2.1 AA (Q5-A)  
**Platform**: Desktop only (Q6-A)

---

## 1. Perceivable

### 1.1 Text Alternatives

- [ ] All non-text content has text alternatives
  - [ ] Icons have `aria-label` or `title` attribute
  - [ ] Charts have descriptive text
  - [ ] Images have `alt` text
  - [ ] Emojis in Telegram bot are accompanied by text

### 1.2 Time-Based Media

- [ ] N/A (no audio/video content in dashboard)

### 1.3 Adaptable

- [ ] Content can be presented in different ways without losing information
  - [ ] Tables have row and column headers
  - [ ] Linear reading order is logical
  - [ ] DOM order matches visual order

### 1.4 Distinguishable

- [ ] Text and images of text have sufficient contrast
  - [ ] Normal text: contrast ratio ≥ 4.5:1
  - [ ] Large text (18pt+): contrast ratio ≥ 3:1
  - [ ] UI components and graphical objects: contrast ratio ≥ 3:1
- [ ] Text can be resized up to 200% without loss of content
- [ ] Images of text are avoided, or used decoratively

---

## 2. Operable

### 2.1 Keyboard Accessible

- [ ] All functionality is available via keyboard
  - [ ] Tab order follows logical sequence
  - [ ] All interactive elements are focusable
  - [ ] Focus indicator is visible (2px minimum)
  - [ ] No keyboard trap
  - [ ] Escape key closes modals
  - [ ] Enter key activates focused buttons
- [ ] Keyboard focus does not get lost
  - [ ] Focus moves predictably when modals open/close
  - [ ] Focus returns to triggering element after modal close

### 2.2 Enough Time

- [ ] N/A (no time-limited content)
- [ ] Moving, blinking, or scrolling content can be paused
  - [ ] Auto-refresh is manual (Q1-B), not automatic

### 2.3 Seizures and Physical Reactions

- [ ] No content flashes more than 3 times per second
- [ ] N/A (no flashing content in dashboard)

### 2.4 Navigable

- [ ] Ways to navigate the page are provided
  - [ ] Skip to main content link (optional for desktop)
  - [ ] Page titles are descriptive
  - [ ] Headings are hierarchical (h1 → h2 → h3)
  - [ ] Landmarks are used (header, main, nav, aside, footer)
- [ ] Link purpose is clear from context
  - [ ] Button labels are descriptive
  - [ ] Links have meaningful text (not "click here")

### 2.5 Input Modalities

- [ ] Functions can be operated by mouse
- [ ] Functions can be operated by keyboard
- [ ] Touch targets are at least 44x44px (for touch devices, if tablet support added)

---

## 3. Understandable

### 3.1 Readable

- [ ] Text content is readable and understandable
  - [ ] Language of page is identified (`lang="pt-BR"`)
  - [ ] Abbreviations are defined on first use
  - [ ] Technical terms are explained
- [ ] Text spacing and layout can be adjusted
  - [ ] Line height is at least 1.5 times font size
  - [ ] Paragraph spacing is at least 2 times font size
  - [ ] Character spacing is at least 0.12 times font size

### 3.2 Predictable

- [ ] Web pages appear and operate in predictable ways
  - [ ] Navigation is consistent across pages
  - [ ] Identical elements have consistent behavior
  - [ ] Consistent identification (icons, labels)
- [ ] Input assistance is provided
  - [ ] Form fields have labels
  - [ ] Required fields are marked
  - [ ] Error messages are clear and specific
  - [ ] Validation happens on submit (or inline)

### 3.3 Input Assistance

- [ ] Errors are identified and described
  - [ ] Form errors are announced to screen readers
  - [ ] Error messages are associated with the field
  - [ ] Errors are in red with high contrast
- [ ] Labels or instructions are provided
  - [ ] All form inputs have visible labels
  - [ ] Instructions are provided before submission
- [ ] Error prevention is provided
  - [ ] Confirmation for destructive actions
  - [ ] Reversible actions where possible

---

## 4. Robust

### 4.1 Compatible

- [ ] Content is compatible with current and future user agents
  - [ ] Valid HTML
  - [ ] ARIA roles are used correctly
  - [ ] Name, role, value are set for custom components
  - [ ] Status messages are announced (`role="alert"`)
- [ ] ARIA landmarks are used
  - [ ] `role="banner"` for header
  - [ ] `role="main"` for main content
  - [ ] `role="navigation"` for navigation
  - [ ] `role="complementary"` for sidebar/alerts
  - [ ] `role="dialog"` for modals

---

## 5. Dashboard-Specific Checklist

### 5.1 Visão Geral (M1)

- [ ] H1: "W Levitt · Agente SDR"
- [ ] Landmarks: header, main, aside (alerts)
- [ ] Filters are keyboard accessible (tab through dropdowns)
- [ ] Refresh button has focus indicator
- [ ] KPI cards have semantic structure
- [ ] Kanban columns have heading (h2)
- [ ] "Ver →" buttons have `aria-label` (ex: "Ver leads Novos")
- [ ] Alert items have `role="alert"` for screen readers
- [ ] Filter state is announced when changed

### 5.2 Detalhe do Lead (M2)

- [ ] H2: "Lead #4821 · Empresa XYZ"
- [ ] Landmarks: main, section
- [ ] Back button has `aria-label="Voltar"`
- [ ] Status dropdown has `aria-label="Status do lead"`
- [ ] Action buttons have `aria-label` (ex: "Atualizar status do lead")
- [ ] Timeline has semantic list structure
- [ ] Timeline milestones have icons + text (not icons alone)
- [ ] Anomaly banner has `role="alert"`

### 5.3 Modals

- [ ] Modals have `role="dialog"`
- [ ] Modals have `aria-modal="true"`
- [ ] Modal title is in dialog heading
- [ ] Close button has `aria-label="Fechar"`
- [ ] Focus is trapped inside modal
- [ ] Escape key closes modal
- [ ] Focus returns to triggering element after close
- [ ] Form inputs have visible labels
- [ ] Error messages have `role="alert"`

### 5.4 Telegram Bot

- [ ] Text is clear and not color-dependent
- [ ] Emojis are accompanied by text
- [ ] Consent message is clear and actionable
- [ ] Voice transcription text is readable
- [ ] Inline buttons have clear labels

---

## 6. Screen Reader Testing

### 6.1 NVDA / JAWS / VoiceOver

- [ ] Page title is announced
- [ ] Landmarks are announced
- [ ] Tab order follows visual order
- [ ] Focus indicator is announced
- [ ] Button labels are announced
- [ ] Form labels are associated with inputs
- [ ] Error messages are announced
- [ ] Modal open/close is announced
- [ ] Dynamic content updates are announced (toasts)

### 6.2 Keyboard Navigation

- [ ] Tab through entire page works
- [ ] Shift+Tab reverses direction
- [ ] Enter activates focused button
- [ ] Escape closes modal
- [ ] Space toggles checkbox
- [ ] Arrow keys navigate dropdowns
- [ ] Focus never gets lost

---

## 7. Color Contrast Testing

### 7.1 Contrast Ratios

- [ ] Primary (#4F8EF7) on white (#FFFFFF): 4.5:1 ✓
- [ ] Text (#262730) on white (#FFFFFF): 12.6:1 ✓
- [ ] Success (#00C851) on white: 4.8:1 ✓
- [ ] Warning (#FFBB33) on white: 2.1:1 ✗ (need darker yellow)
- [ ] Error (#FF4444) on white: 4.5:1 ✓
- [ ] Border (#E0E0E0) on white: 1.4:1 ✗ (borders are decorative, not text)

**Fix needed**: Warning color (#FFBB33) needs to be darker for WCAG AA (≥ 4.5:1). Consider #FF9900 or similar.

---

## 8. Testing Tools

- [ ] Axe DevTools (Chrome extension)
- [ ] WAVE (web accessibility evaluation tool)
- [ ] Lighthouse (Chrome DevTools)
- [ ] NVDA / JAWS / VoiceOver (screen readers)
- [ ] Keyboard-only navigation

---

## 9. Priority Items for POC

Given the hackathon timeline, prioritize:

1. **Must-have**:
   - [ ] Keyboard navigation (tab order, focus indicator)
   - [ ] Form labels and error messages
   - [ ] Modal focus trap and escape key
   - [ ] Basic contrast (fix warning color)
   - [ ] Screen reader basics (landmarks, ARIA roles)

2. **Nice-to-have** (if time permits):
   - [ ] Skip to main content link
   - [ ] Full ARIA compliance
   - [ ] Comprehensive screen reader testing
   - [ ] Advanced contrast tuning

---

## 10. Notes

- Desktop only (Q6-A) simplifies mobile accessibility requirements
- Telegram bot is third-party app; ensure text clarity and color independence
- MCP Inspector is demo tool; accessibility not critical for POC
- Streamlit default theme has good contrast, but warning color needs adjustment