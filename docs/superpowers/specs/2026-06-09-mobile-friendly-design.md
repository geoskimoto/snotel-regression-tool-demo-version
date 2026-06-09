# Mobile-Friendly Design Spec
**Date:** 2026-06-09  
**Target:** Phone screens (≤767px)  
**Use case:** Full analysis (station selection, model training) + viewing results/maps

---

## Goals

Make the SNOTEL Regression Tool usable on a phone without degrading the existing desktop experience.

---

## Section 1: Layout Architecture

### Constraint
Dash requires unique component IDs. Rendering `get_control_view()` twice (desktop sidebar + mobile offcanvas) would break callbacks. Controls must be rendered **once**.

### Solution
Use `dbc.Offcanvas` as the single container for controls, with `assets/mobile.css` applying Bootstrap's responsive offcanvas pattern:

- **Desktop (md+):** CSS overrides make the offcanvas appear as a static inline sidebar (25% width, no backdrop, no slide animation)
- **Mobile (<md):** True offcanvas drawer — triggered by a "Controls" button in the top bar

### Changes to `app.py`
- Remove `dbc.Col(get_control_view(), width=3)` desktop sidebar
- Add `dbc.Offcanvas(get_control_view(), id="controls-offcanvas", ...)` **inside `dbc.Row`** as a sibling to the main content col — this is required so the desktop CSS (position: relative, width: 25%) participates in the row's flexbox and pushes content
- Add a mobile-only trigger button (`d-md-none`) above the tabs row
- Add callback to toggle `is_open` on `controls-offcanvas`
- The main content column uses `xs=12, md=9`: full-width on mobile (offcanvas is position:fixed, out of flow), 9-cols on desktop (offcanvas sidebar takes the remaining 25%)

### Changes to `views/controls.py`
No structural changes — `get_control_view()` is unchanged except for the date picker replacements (Section 2).

---

## Section 2: Controls Panel Changes

### Date Pickers
Replace both `dbc.InputGroup` + `dcc.DatePickerSingle` pairs with `dbc.Input(type="date")`. Component IDs are preserved so no callback changes are needed.

**Training date range** (`startdate_picker`, `enddate_picker`):
- Before: Two `dcc.DatePickerSingle` inside a `dbc.InputGroup`
- After: Two labeled `dbc.Input(type="date")` stacked vertically

**Prediction date range** (`predict_startdate_picker`, `predict_enddate_picker`):
- Same pattern as above

### Database Management Buttons
The `dbc.ButtonGroup` in `app.py` gets `className="flex-column flex-sm-row w-100"` so buttons stack vertically on phone, inline on sm+.

---

## Section 3: `assets/mobile.css`

Two responsibilities:

### 1. Responsive offcanvas (desktop sidebar behavior)
```css
@media (min-width: 768px) {
    #controls-offcanvas {
        position: relative !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
        width: 25% !important;
        height: auto !important;
        border: none !important;
        background: transparent !important;
        z-index: auto !important;
    }
    #controls-offcanvas .offcanvas-header {
        display: none !important;
    }
}
```

### 2. Plot overflow guard
```css
@media (max-width: 767px) {
    .js-plotly-plot, .plotly {
        max-width: 100vw !important;
    }
}
```

---

## Files Changed

| File | Change |
|---|---|
| `app.py` | Replace sidebar Col with Offcanvas + trigger button + toggle callback; ButtonGroup className |
| `views/controls.py` | Replace `dcc.DatePickerSingle` pairs with `dbc.Input(type="date")` |
| `assets/mobile.css` | New file — responsive offcanvas + plot overflow guard |

---

## Out of Scope

- No changes to callbacks, data logic, or database layer
- No JS build pipeline introduced
- Desktop layout preserved without modification
