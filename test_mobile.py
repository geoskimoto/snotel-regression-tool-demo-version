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
