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


def test_offcanvas_toggle_callback_registered():
    # In Dash 2.x, callback_map keys are stringified output specs like
    # "controls-offcanvas.is_open" or "..controls-offcanvas.is_open.."
    import app as app_module
    found = any(
        'controls-offcanvas' in key and 'is_open' in key
        for key in app_module.app.callback_map
    )
    assert found, "No callback registered with controls-offcanvas.is_open as output"
