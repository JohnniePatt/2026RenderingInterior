from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_spatial_correspondence import evaluation
from test_automatic_correspondence import Detector, candidate, context


def page(root,state):
    from pathlib import Path
    from app.ui.spatial_validation import render
    render(Path(root),state)


def test_automatic_run_and_review_without_clicks(evaluation,monkeypatch):
    from app.ui import spatial_validation as ui
    root,state=evaluation; state['cameras']={'camera_001':{}}
    c=context(evaluation); detector=Detector([candidate(c['ids']==1)])
    monkeypatch.setattr(ui,'get_backend',lambda config:detector)
    app=AppTest.from_function(page,kwargs={'root':str(root),'state':state}).run(timeout=15)
    assert not app.exception
    next(w for w in app.selectbox if w.label=='Generated Render').select('render_a.png').run()
    next(w for w in app.button if w.label=='Run Automatic Correspondence').click().run()
    assert not app.exception and not app.error
    assert detector.calls==['furniture']  # legacy semantic fallback, no invented object class
    assert context(evaluation)['doc']['correspondences'][0]['status']=='auto_matched'
    next(w for w in app.checkbox if w.label=='Show cases needing attention only').uncheck().run()
    next(w for w in app.button if w.label=='Confirm').click().run()
    assert not app.exception
    assert context(evaluation)['doc']['correspondences'][0]['status']=='human_verified'


def test_missing_model_shows_error_without_false_detection(evaluation,monkeypatch):
    from app.ui import spatial_validation as ui
    root,state=evaluation; state['cameras']={'camera_001':{}}
    def unavailable(config): raise ValueError('SAM 3 weights are not installed')
    monkeypatch.setattr(ui,'get_backend',unavailable)
    app=AppTest.from_function(page,kwargs={'root':str(root),'state':state}).run(timeout=15)
    next(w for w in app.button if w.label=='Run Automatic Correspondence').click().run()
    assert not app.exception
    assert any('not installed' in error.value for error in app.error)
    assert context(evaluation)['doc'] is None
