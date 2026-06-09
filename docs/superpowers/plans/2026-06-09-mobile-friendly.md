# Mobile-Friendly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SNOTEL Regression Tool fully usable on phone screens without changing the desktop experience.

**Architecture:** Add `assets/mobile.css` to make `dbc.Offcanvas` behave as a static sidebar on desktop via CSS overrides; render the controls panel once inside the offcanvas (inside the main `dbc.Row`); replace `dcc.DatePickerSingle` with `dbc.Input(type="date")` and update all callback property references from `"date"` to `"value"`; add a mobile-only trigger button and toggle callback.

**Tech Stack:** Dash 2.18.2, dash-bootstrap-components, Bootstrap 5

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `assets/mobile.css` | Create (+ mkdir) | Responsive offcanvas sidebar + plot overflow guard |
| `views/controls.py` | Modify | Replace `dcc.DatePickerSingle` with `dbc.Input(type="date")` |
| `app.py` | Modify | Callback property refs, layout restructure, toggle callback |
| `test_mobile.py` | Create | Layout structure and callback property tests |

---

### Task 1: Create `assets/mobile.css`

**Files:**
- Create: `assets/mobile.css`

- [ ] **Step 1: Create the assets directory and CSS file**

```bash
cd /home/snotel/htdocs/app
sudo -u snotel mkdir -p assets
sudo -u snotel tee assets/mobile.css > /dev/null << 'EOF'
/* Desktop (md+): make offcanvas appear as a static inline sidebar */
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

/* Mobile: prevent Plotly graphs from overflowing viewport */
@media (max-width: 767px) {
    .js-plotly-plot, .plotly {
        max-width: 100vw !important;
    }
}
EOF
```

- [ ] **Step 2: Verify the file was created with correct content**

```bash
cat /home/snotel/htdocs/app/assets/mobile.css
```
Expected: Two `@media` blocks — one for `min-width: 768px` and one for `max-width: 767px`.

- [ ] **Step 3: Commit**

```bash
cd /home/snotel/htdocs/app
sudo -u snotel git add assets/mobile.css
sudo -u snotel git commit -m "feat: add mobile.css with responsive offcanvas sidebar and plot overflow guard"
```

---

### Task 2: Replace date pickers in `views/controls.py`

**Files:**
- Modify: `views/controls.py`
- Create: `test_mobile.py`

**Context:** `dcc.DatePickerSingle` exposes its selected date via the `date` property. `dbc.Input(type="date")` uses the `value` property and expects a `"YYYY-MM-DD"` string. The `html_for` attribute on `dbc.Input` uses the HTML `min` attribute (not `min_date_allowed`). Callback property updates happen in Task 3 — this task is layout only.

- [ ] **Step 1: Write the failing test**

Create `test_mobile.py` in the project root (`/home/snotel/htdocs/app/test_mobile.py`):

```python
import sys
sys.path.insert(0, '/home/snotel/htdocs/app')

import dash_bootstrap_components as dbc
from views.controls import get_control_view


def find_by_id(node, target_id):
    if getattr(node, 'id', None) == target_id:
        return node
    children = getattr(node, 'children', None)
    if not children:
        return None
    items = children if isinstance(children, list) else [children]
    for child in items:
        result = find_by_id(child, target_id)
        if result is not None:
            return result
    return None


def find_all_by_id(node, target_id, results=None):
    if results is None:
        results = []
    if getattr(node, 'id', None) == target_id:
        results.append(node)
    children = getattr(node, 'children', None)
    if children:
        items = children if isinstance(children, list) else [children]
        for child in items:
            find_all_by_id(child, target_id, results)
    return results


APP_PY = '/home/snotel/htdocs/app/app.py'


def test_date_pickers_are_native_inputs():
    layout = get_control_view()
    for picker_id in (
        'startdate_picker', 'enddate_picker',
        'predict_startdate_picker', 'predict_enddate_picker',
    ):
        component = find_by_id(layout, picker_id)
        assert component is not None, f"Component {picker_id} not found in controls layout"
        assert isinstance(component, dbc.Input), (
            f"{picker_id} should be dbc.Input, got {type(component).__name__}"
        )
        assert component.type == 'date', (
            f"{picker_id} should have type='date', got '{component.type}'"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_date_pickers_are_native_inputs -v
```
Expected: FAIL — components are still `dcc.DatePickerSingle`

- [ ] **Step 3: Replace the training date range block in `views/controls.py`**

Find this block (around line 160):
```python
dbc.InputGroup(
    id="train-model-date-grp",
    children=[
        dcc.DatePickerSingle(
            id="startdate_picker",
            min_date_allowed=date(1950, 10, 1),
            date=newest_begin_date,
        ),
        dcc.DatePickerSingle(
            id="enddate_picker",
            min_date_allowed=date(1950, 10, 1),
            date=date.today() - relativedelta(days=15),
        ),
    ],
),
```

Replace with:
```python
html.Div(
    id="train-model-date-grp",
    children=[
        dbc.Label("Start:", html_for="startdate_picker"),
        dbc.Input(
            id="startdate_picker",
            type="date",
            min="1950-10-01",
            value=str(newest_begin_date),
        ),
        dbc.Label("End:", html_for="enddate_picker"),
        dbc.Input(
            id="enddate_picker",
            type="date",
            min="1950-10-01",
            value=str(date.today() - relativedelta(days=15)),
        ),
    ],
),
```

- [ ] **Step 4: Replace the prediction date range block in `views/controls.py`**

Find this block (around line 200):
```python
dbc.InputGroup(
    id="run-model-date-grp",
    children=[
        dcc.DatePickerSingle(
            id="predict_startdate_picker",
            date=date.today() - relativedelta(months=1),
        ),
        dcc.DatePickerSingle(
            id="predict_enddate_picker",
            date=date.today(),
        ),
    ],
),
```

Replace with:
```python
html.Div(
    id="run-model-date-grp",
    children=[
        dbc.Label("Start:", html_for="predict_startdate_picker"),
        dbc.Input(
            id="predict_startdate_picker",
            type="date",
            value=str(date.today() - relativedelta(months=1)),
        ),
        dbc.Label("End:", html_for="predict_enddate_picker"),
        dbc.Input(
            id="predict_enddate_picker",
            type="date",
            value=str(date.today()),
        ),
    ],
),
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_date_pickers_are_native_inputs -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
sudo -u snotel git add views/controls.py test_mobile.py
sudo -u snotel git commit -m "feat: replace dcc.DatePickerSingle with native dbc.Input date pickers"
```

---

### Task 3: Update callback property references in `app.py`

**Files:**
- Modify: `app.py`

**Context:** All `State` and `Output` references to the four date picker component IDs must change from property `"date"` to `"value"`. There are three callbacks affected:
- `populate_dropdowns`: one `Output("startdate_picker", "date")` → must also format the returned value as `"YYYY-MM-DD"` string (was a `datetime` object, `dbc.Input` needs a string)
- `train_test_figures`: `State("startdate_picker", "date")`, `State("enddate_picker", "date")`
- `train_pred_figures`: all four date picker States
- `save_model`: `State("startdate_picker", "date")`, `State("enddate_picker", "date")`

- [ ] **Step 1: Write the failing test**

Add this test to `test_mobile.py`:

```python
def test_date_picker_callbacks_use_value_not_date():
    """Check app.py source directly — no fragile callback_map introspection."""
    with open(APP_PY) as f:
        content = f.read()
    for picker_id in ('startdate_picker', 'enddate_picker',
                      'predict_startdate_picker', 'predict_enddate_picker'):
        old_ref_double = f'"{picker_id}", "date"'
        old_ref_single = f"'{picker_id}', 'date'"
        assert old_ref_double not in content and old_ref_single not in content, (
            f"Found old .date property reference for {picker_id} in app.py — "
            f"should be .value"
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_date_picker_callbacks_use_value_not_date -v
```
Expected: FAIL — `"date"` property references still present in `app.py`

- [ ] **Step 3: Update `Output("startdate_picker", "date")` in `populate_dropdowns` callback decorator**

In `app.py`, in the `@app.callback(...)` decorator for `populate_dropdowns`, change:
```python
Output("startdate_picker", "date"),
```
to:
```python
Output("startdate_picker", "value"),
```

- [ ] **Step 4: Update the return value in `populate_dropdowns` function body**

In the `populate_dropdowns` function, the last item in the `return (...)` tuple is `max_begin_date` — a `datetime` or `date` object. Change it to a formatted string:
```python
max_begin_date.strftime("%Y-%m-%d"),
```

- [ ] **Step 5: Update `State` references in `train_test_figures`**

In the `@app.callback(...)` decorator for `train_test_figures`, change:
```python
State("startdate_picker", "date"),
State("enddate_picker", "date"),
```
to:
```python
State("startdate_picker", "value"),
State("enddate_picker", "value"),
```

- [ ] **Step 6: Update `State` references in `train_pred_figures`**

In the `@app.callback(...)` decorator for `train_pred_figures`, change:
```python
State("startdate_picker", "date"),
State("enddate_picker", "date"),
State("predict_startdate_picker", "date"),
State("predict_enddate_picker", "date"),
```
to:
```python
State("startdate_picker", "value"),
State("enddate_picker", "value"),
State("predict_startdate_picker", "value"),
State("predict_enddate_picker", "value"),
```

- [ ] **Step 7: Update `State` references in `save_model`**

In the `@app.callback(...)` decorator for `save_model`, change:
```python
State("startdate_picker", "date"),
State("enddate_picker", "date"),
```
to:
```python
State("startdate_picker", "value"),
State("enddate_picker", "value"),
```

- [ ] **Step 8: Run test to verify it passes**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_date_picker_callbacks_use_value_not_date -v
```
Expected: PASS

- [ ] **Step 9: Run all tests**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py test_regression.py -v
```
Expected: All pass

- [ ] **Step 10: Commit**

```bash
sudo -u snotel git add app.py
sudo -u snotel git commit -m "fix: update date picker callback properties from 'date' to 'value' for dbc.Input"
```

---

### Task 4: Restructure `app.py` layout — offcanvas, trigger button, ButtonGroup

**Files:**
- Modify: `app.py`

**Context:** Replace the `dbc.Col(get_control_view(), width=3)` sidebar with `dbc.Offcanvas(get_control_view(), id="controls-offcanvas", ...)` placed as the **first child** of the main `dbc.Row` so the desktop CSS override (position: relative, width: 25%) participates in flexbox and acts as a sidebar. Add a mobile-only trigger button in a row above. Change the main content col to `xs=12, md=9`. Update the `dbc.ButtonGroup` className.

The `dbc.Offcanvas` must be imported — check the existing `import dash_bootstrap_components as dbc` covers it.

- [ ] **Step 1: Write the failing tests**

Add to `test_mobile.py` (uses `find_by_id` and `find_all_by_id` helpers already defined at the top of the file):

```python
def test_layout_has_offcanvas():
    import app as app_module
    layout = app_module.app.layout
    offcanvas = find_by_id(layout, 'controls-offcanvas')
    assert offcanvas is not None, "controls-offcanvas not found in layout"


def test_layout_has_mobile_trigger_button():
    import app as app_module
    layout = app_module.app.layout
    btn = find_by_id(layout, 'controls-offcanvas-toggle')
    assert btn is not None, "controls-offcanvas-toggle button not found in layout"


def test_controls_rendered_exactly_once():
    import app as app_module
    layout = app_module.app.layout
    cards = find_all_by_id(layout, 'input-card')
    assert len(cards) == 1, f"input-card should appear exactly once, found {len(cards)}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_layout_has_offcanvas test_mobile.py::test_layout_has_mobile_trigger_button -v
```
Expected: FAIL

- [ ] **Step 3: Replace the layout row in `app.py`**

Find the `dbc.Row` block that starts with `dbc.Col(get_control_view(), width=3)` (around line 40). Replace the **entire** `dbc.Row(...)` containing the sidebar and tabs with:

```python
dbc.Row(
    [
        dbc.Button(
            "Controls",
            id="controls-offcanvas-toggle",
            color="secondary",
            size="sm",
            className="d-md-none mb-1",
            n_clicks=0,
        ),
    ],
    className="my-1",
),
dbc.Row(
    [
        dbc.Offcanvas(
            get_control_view(),
            id="controls-offcanvas",
            title="Controls",
            is_open=False,
            placement="start",
            backdrop=True,
        ),
        dbc.Col(
            children=[
                dbc.Tabs(
                    children=[
                        dbc.Tab(
                            label="Training",
                            children=[
                                dbc.Row(
                                    dbc.Col(html.Div(id="traintest-plots")),
                                ),
                                dbc.Row(
                                    dbc.Col(html.Div(id="modelfit-plots")),
                                ),
                            ],
                        ),
                        dbc.Tab(
                            label="Map",
                            children=[
                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                            id="station-map", className="p-2"
                                        ),
                                    ),
                                ),
                            ],
                        ),
                        dbc.Tab(
                            label="Results",
                            children=[
                                dbc.Row(
                                    dbc.Col(html.Div(id="pred-plots")),
                                ),
                            ],
                        ),
                        dbc.Tab(
                            label="Database Management",
                            children=[
                                dbc.Row(
                                    dbc.Col(
                                        children=[
                                            html.Br(),
                                            dbc.ButtonGroup(
                                                [
                                                    dbc.Button(
                                                        id="view-datatable-button",
                                                        children="View Database",
                                                        color="secondary",
                                                    ),
                                                    dbc.Button(
                                                        id="save-to-db-button",
                                                        children="Save Model to DB",
                                                        color="secondary",
                                                    ),
                                                    dbc.Button(
                                                        id='update-db-button',
                                                        children="Save changes to DB",
                                                        color="secondary",
                                                    ),
                                                ],
                                                className="flex-column flex-sm-row w-100",
                                            ),
                                            html.Div(
                                                id="save-message", children=[]
                                            ),
                                            html.Div(
                                                id="update-message", children=[]
                                            ),
                                            html.Div(
                                                id="view-datatable", children=[]
                                            ),
                                        ]
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            xs=12,
            md=9,
        ),
    ],
    className="my-2",
),
```

- [ ] **Step 4: Run layout tests**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_layout_has_offcanvas test_mobile.py::test_layout_has_mobile_trigger_button test_mobile.py::test_controls_rendered_exactly_once -v
```
Expected: All PASS

- [ ] **Step 5: Run full test suite**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py test_regression.py -v
```
Expected: All pass

- [ ] **Step 6: Commit**

```bash
sudo -u snotel git add app.py test_mobile.py
sudo -u snotel git commit -m "feat: replace sidebar col with responsive offcanvas, add mobile trigger button, stack DB buttons on mobile"
```

---

### Task 5: Add offcanvas toggle callback and deploy

**Files:**
- Modify: `app.py`

**Context:** The `controls-offcanvas-toggle` button needs a callback to toggle `is_open` on `controls-offcanvas`. On desktop, the CSS makes the offcanvas always visible regardless of `is_open`, so this callback only has meaningful effect on mobile. Place the callback immediately after `populate_dropdowns`.

- [ ] **Step 1: Write the failing test**

Add to `test_mobile.py`:

```python
def test_offcanvas_toggle_callback_registered():
    # In Dash 2.x, callback_map keys are stringified output specs like
    # "controls-offcanvas.is_open" or "..controls-offcanvas.is_open.."
    import app as app_module
    found = any(
        'controls-offcanvas' in key and 'is_open' in key
        for key in app_module.app.callback_map
    )
    assert found, "No callback registered with controls-offcanvas.is_open as output"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_offcanvas_toggle_callback_registered -v
```
Expected: FAIL

- [ ] **Step 3: Add the toggle callback to `app.py`**

After the `populate_dropdowns` callback function (before `train_test_figures`), add:

```python
@app.callback(
    Output("controls-offcanvas", "is_open"),
    Input("controls-offcanvas-toggle", "n_clicks"),
    State("controls-offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggle_controls_offcanvas(n_clicks, is_open):
    return not is_open
```

- [ ] **Step 4: Run toggle callback test**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py::test_offcanvas_toggle_callback_registered -v
```
Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
cd /home/snotel/htdocs/app && sudo -u snotel .venv/bin/pytest test_mobile.py test_regression.py -v
```
Expected: All pass

- [ ] **Step 6: Commit**

```bash
sudo -u snotel git add app.py test_mobile.py
sudo -u snotel git commit -m "feat: add offcanvas toggle callback for mobile controls drawer"
```

- [ ] **Step 7: Deploy and verify**

```bash
sudo systemctl restart snotel-regression.service
sudo systemctl status snotel-regression.service
```
Expected: `Active: active (running)`
