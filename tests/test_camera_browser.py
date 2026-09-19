"""End-to-end live camera interaction against real Streamlit and WebGL."""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import ezdxf
import numpy as np
import pytest
from app.layout.manager import create
from app.cad.entities import load_for_editor
from app.annotation.tagging import apply

pytestmark = pytest.mark.skipif(os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Opt-in browser acceptance")
REPO = Path(__file__).resolve().parents[1]


def room(workspace):
    doc = ezdxf.new("R2010")
    doc.units = 4
    model = doc.modelspace()
    model.add_lwpolyline([(0,0),(6000,0),(6000,5000),(0,5000)],close=True)
    model.add_lwpolyline([(0,0),(0,5000),(6000,5000),(6000,0)])
    model.add_lwpolyline([(2000,2200),(3500,2200),(3500,3200),(2000,3200)],close=True)
    model.add_lwpolyline([(500,3700),(2300,3700),(2300,4500),(500,4500)],close=True)
    stream = io.StringIO();doc.write(stream)
    root, _ = create(workspace,"Camera acceptance room","room.dxf",stream.getvalue().encode())
    state, geometry, _ = load_for_editor(root)
    for entity, semantic, props in zip(geometry, ["floor","wall","furniture","furniture"], [
        {"elevation":0,"thickness":150}, {"height":2800,"base_elevation":0},
        {"category":"dining_table","height":750}, {"category":"sofa","height":820}]):
        state = apply(root,state,[entity["id"]],semantic,props)
    return root


def test_camera_live_drag_multiple_restore_and_projection(tmp_path):
    from playwright.sync_api import sync_playwright, expect
    root = room(tmp_path / "workspace")
    path = root / "layout.json"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    url=f"http://127.0.0.1:{port}"
    def start():
        with (tmp_path / "server.log").open("a") as log:
            proc=subprocess.Popen([sys.executable,"-m","streamlit","run","app.py","--server.headless=true",
                f"--server.port={port}","--server.address=127.0.0.1"], cwd=REPO,
                env=dict(os.environ,CAD_WORKSPACE=str(root.parent)),stdout=log,stderr=log)
        for _ in range(100):
            try:
                urllib.request.urlopen(url+"/_stcore/health",timeout=1)
                return proc
            except OSError:time.sleep(.1)
        proc.terminate();raise RuntimeError((tmp_path/"server.log").read_text())
    proc=start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=["--no-sandbox","--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
            page=browser.new_page(viewport={"width":1700,"height":1200})
            errors=[];page.on("pageerror",lambda error:errors.append(str(error)))
            def open_camera_mode():
                page.goto(url)
                page.get_by_role("button",name="Open / Edit Layout",exact=True).click()
                page.get_by_text("Camera Mode",exact=True).click()
                frame=page.frame_locator('iframe[title="app.ui.viewer.cad_viewer"]').last
                expect(frame.locator('#camera-workspace')).to_be_visible()
                return frame
            frame=open_camera_mode()
            workspace=frame.locator('#camera-workspace')
            canvas=frame.locator('#camera-plan canvas')
            preview=frame.locator('#perspective-view canvas')
            def state():return json.loads(path.read_text())
            def saved(count):
                expect(frame.locator('#camera-save')).to_have_text('Saved',timeout=10000)
                assert len(state()["cameras"])==count
            def pixel(x,y):
                v=json.loads(workspace.get_attribute('data-view'))
                return {'x':((x-v['origin']['x']-v['center']['x'])/(v['viewHeight']*v['width']/v['height'])+.5)*v['width'],
                        'y':(.5-(y-v['origin']['y']-v['center']['y'])/v['viewHeight'])*v['height']}
            def drag(a,b,check_live=False):
                canvas.scroll_into_view_if_needed()
                bounds=canvas.bounding_box()
                before=path.read_bytes()
                page.mouse.move(bounds['x']+a['x'],bounds['y']+a['y']);page.mouse.down()
                first=preview.screenshot() if check_live else None
                page.mouse.move(bounds['x']+b['x'],bounds['y']+b['y'],steps=12)
                if check_live:
                    assert preview.screenshot()!=first
                    assert path.read_bytes()==before, "Dragging must not write every frame"
                page.mouse.up()
            frame.get_by_role('button',name='+ Add Camera',exact=True).click()
            drag(pixel(3000,800),pixel(3000,1800),check_live=True)
            saved(1)
            first_id=state()['active_camera']
            first=state()['cameras'][first_id]
            assert first['position']==pytest.approx([3000,800,1500],abs=15)
            assert first['heading_deg']==pytest.approx(90,abs=1)
            expect(frame.locator('#perspective-status')).to_contain_text('proxy parts')
            # Edit controls inside the component: input updates live, change commits.
            for label,value in [('Camera name','Main view'),('Camera height','1650'),('Camera pitch','-8'),('Horizontal FOV','75')]:
                control=frame.get_by_label(label,exact=True)
                control.fill(value);control.press('Tab');saved(1)
            camera=state()['cameras'][first_id]
            assert camera['name']=='Main view' and camera['position'][2]==1650
            assert camera['pitch_deg']==-8 and camera['fov_deg']==75
            # Position drag and heading-handle drag must both change actual perspective live.
            drag(pixel(*camera['position'][:2]),pixel(3800,1100),check_live=True);saved(1)
            dot=pixel(*state()['cameras'][first_id]['position'][:2])
            drag({'x':dot['x'],'y':dot['y']-85},{'x':dot['x']-60,'y':dot['y']-60},check_live=True);saved(1)
            assert state()['cameras'][first_id]['heading_deg']==pytest.approx(135,abs=2)
            # Actual viewport aspect changes K consistently without saving camera state.
            projection=workspace.get_attribute('data-projection');previous=path.read_bytes()
            page.set_viewport_size({'width':1450,'height':1200})
            expect(workspace).not_to_have_attribute('data-projection',projection)
            assert path.read_bytes()==previous
            frame.get_by_role('button',name='+ Add Camera',exact=True).click()
            drag(pixel(1000,1000),pixel(2200,2300));saved(2)
            second_id=state()['active_camera'];assert second_id!=first_id
            frame.get_by_label('Active Camera',exact=True).select_option(first_id);saved(2)
            assert state()['active_camera']==first_id
            expected=state()['cameras']
            # Verify JS projection orientation agrees with research K[R|t].
            result=workspace.evaluate('''async () => {
                const THREE=await import('./vendor/three.module.js');
                const {configurePerspective}=await import('./camera_projection.js');
                const c=new THREE.PerspectiveCamera();
                configurePerspective(c,{position:[0,0,1.5],heading_deg:0,pitch_deg:0,fov_deg:60},800/600,{x:0,y:0},null,.001);
                return [[10,0,1.5],[10,-1,1.5],[10,0,2.5]].map(p=>{
                    const q=new THREE.Vector3(...p).project(c);return [(q.x+1)*400,(1-q.y)*300];
                });
            }''')
            from app.geometry.camera import project_points,new_camera
            expected_pixels,_=project_points([[10,0,1.5],[10,-1,1.5],[10,0,2.5]],new_camera(0,0,'m'),800,600)
            np.testing.assert_allclose(result,expected_pixels,atol=1e-8)
            # Verify the renderer extrudes source footprints to the same metric bounds as Python.
            mesh_bounds=workspace.evaluate('''async () => {
                const THREE=await import('./vendor/three.module.js');
                const {buildProxy,disposeGroup}=await import('./proxy.js');
                const group=buildProxy([{entity_id:'table',semantic:'furniture',kind:'solid',
                    outer:[[10,20],[110,20],[110,70],[10,70],[10,20]],holes:[],z_min:100,z_max:850}],{x:0,y:0});
                const box=new THREE.Box3().setFromObject(group);
                const bounds=[box.min.toArray(),box.max.toArray()];disposeGroup(group);return bounds;
            }''')
            assert mesh_bounds==[[10,20,100],[110,70,850]]
            preview_before_restart=preview.screenshot()
            (REPO/'test-results').mkdir(exist_ok=True)
            page.locator('[data-testid="stMain"]').evaluate('(el)=>el.scrollTop=0')
            page.screenshot(path=str(REPO/'test-results/camera-mode.png'),full_page=True)
            proc.terminate();proc.wait(timeout=10);proc=start()
            page.close();page=browser.new_page(viewport={'width':1450,'height':1200})
            frame=open_camera_mode();workspace=frame.locator('#camera-workspace')
            expect(workspace).to_have_attribute('data-active-camera',first_id)
            assert state()['cameras']==expected
            assert json.loads(workspace.get_attribute('data-camera-state'))==expected[first_id]
            assert frame.locator('#perspective-view canvas').screenshot()==preview_before_restart
            frame.get_by_label('Active Camera',exact=True).select_option(second_id);saved(2)
            assert state()['active_camera']==second_id
            frame.get_by_role('button',name='Delete Camera',exact=True).click();saved(1)
            assert state()['active_camera']==first_id
            assert not errors
            browser.close()
    finally:
        proc.terminate();proc.wait(timeout=10)
