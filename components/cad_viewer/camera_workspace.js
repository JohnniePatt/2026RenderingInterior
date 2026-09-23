import * as THREE from './vendor/three.module.js';
import {boundsOf,buildLines,colors} from './geometry.js';
import {buildProxy,disposeGroup} from './proxy.js';
import {configurePerspective} from './camera_projection.js';
import {applyPass,buildInstanceMapping,renderOffscreenPasses} from './conditioning.js';
import {renderPlanarCanvas} from './homography.js';

// Client-side camera interaction, rendering and a one-in-flight/coalescing save bridge.
// Pointermove and resize never post to Python. Only completed gestures/edits persist.
export function createCameraWorkspace(send) {
  const root=document.createElement('section');root.id='camera-workspace';root.hidden=true;
  root.innerHTML=`<div class="camera-toolbar"><strong>CAMERA MODE</strong>
    <label>Active Camera <select id="active-camera" aria-label="Active Camera"></select></label>
    <button id="add-camera">+ Add Camera</button><button id="delete-camera">Delete Camera</button>
    <span id="camera-save" role="status">Saved</span></div>
    <div class="split-view">
      <div class="plan-pane">
        <div id="camera-plan" class="camera-view">
          <div class="view-title">2D PLAN <button id="camera-fit">Fit plan</button></div>
          <svg id="camera-markers" aria-label="Plan camera markers"></svg>
          <div id="plan-hint" class="view-hint"></div>
        </div>
        <div class="camera-controls">
          <label>Camera name<input id="camera-name" aria-label="Camera name" type="text" maxlength="200"></label>
          <label>Height <span id="height-unit"></span><input id="camera-height" aria-label="Camera height" type="number" step="any"></label>
          <label>Heading °<input id="camera-heading" aria-label="Camera heading" type="number" step="1"></label>
          <label>Pitch °<input id="camera-pitch" aria-label="Camera pitch" type="number" min="-89" max="89" step="1"></label>
          <label>Horizontal FOV °<input id="camera-fov" aria-label="Horizontal FOV" type="number" min="1" max="179" step="1"></label>
          <label>Aspect ratio
            <select id="camera-aspect-preset" aria-label="Aspect ratio preset">
              <option value="1:1">1:1 Square (1024×1024)</option>
              <option value="4:3">4:3 Standard (1024×768)</option>
              <option value="3:2">3:2 Classic (1200×800)</option>
              <option value="16:9">16:9 Gemini AI (1376×768)</option>
              <option value="16:9-fhd">16:9 Full HD (1920×1080)</option>
              <option value="custom">Custom</option>
            </select>
          </label>
          <label>Output W<input id="camera-out-w" aria-label="Output width" type="number" min="64" max="8192" step="1"></label>
          <label>Output H<input id="camera-out-h" aria-label="Output height" type="number" min="64" max="8192" step="1"></label>
        </div>
        <div id="camera-message" role="status"></div>
      </div>
      <div class="preview-pane">
        <div id="perspective-view" class="camera-view">
          <div class="view-title perspective-header">
            <span class="perspective-title-text">ACTIVE CAMERA PERSPECTIVE</span>
            <div class="pass-tabs" role="tablist" aria-label="Conditioning Passes">
              <button class="pass-tab active" data-pass="proxy" type="button" role="tab" aria-selected="true">Proxy</button>
              <button class="pass-tab" data-pass="depth" type="button" role="tab" aria-selected="false">Depth</button>
              <button class="pass-tab" data-pass="instance" type="button" role="tab" aria-selected="false">Instance</button>
              <button class="pass-tab" data-pass="semantic" type="button" role="tab" aria-selected="false">Semantic</button>
              <button class="pass-tab" data-pass="planar" type="button" role="tab" aria-selected="false">Planar</button>
            </div>
            <button id="export-outputs" class="export-btn" type="button" title="Export pixel-aligned conditioning outputs">Export All Outputs</button>
          </div>
          <div id="perspective-empty"></div>
          <div id="perspective-status" class="view-hint"></div>
        </div>
      </div>
    </div>`;
  document.body.append(root);
  const get=id=>document.getElementById(id)||root.querySelector('#'+id);
  const planHost=get('camera-plan'), previewHost=get('perspective-view');
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x111b25);
  const planCamera=new THREE.OrthographicCamera(-1,1,1,-1,.1,100);
  planCamera.position.z=10;planCamera.layers.set(0);
  const perspective=new THREE.PerspectiveCamera();perspective.layers.set(1);
  const ambient=new THREE.HemisphereLight(0xf4f7ff,0x65707a,2.2);ambient.position.set(0,0,1);ambient.layers.enable(1);scene.add(ambient);
  const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(-.5,-.8,1);light.layers.enable(1);scene.add(light);
  const planRenderer=new THREE.WebGLRenderer({antialias:true});
  const previewRenderer=new THREE.WebGLRenderer({antialias:true});
  for(const [renderer,host,label] of [[planRenderer,planHost,'Camera placement plan'],[previewRenderer,previewHost,'Active camera perspective']]) {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.domElement.setAttribute('aria-label',label);
    host.prepend(renderer.domElement);
  }
  const canvas=planRenderer.domElement;
  const planarScene=new THREE.Scene();
  const planarOrthoCamera=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
  const planarQuad=new THREE.Mesh(
    new THREE.PlaneGeometry(2,2),
    new THREE.MeshBasicMaterial({depthTest:false,depthWrite:false})
  );
  planarScene.add(planarQuad);
  let args={},cameras={},active=null,geometry=[],origin={x:0,y:0},lines=null,proxy=null;
  let geometryVersion=null,proxyVersion=null,adding=false,down=null,center={x:0,y:0},viewHeight=100;
  let width=1,height=560,previewWidth=1,previewHeight=560,pending=null,queued=null,revision=null,failed=false,dirty=false;
  let activePass='proxy',exporting=false;
  const clone=value=>JSON.parse(JSON.stringify(value));
  function updateActivePass() {
    const instanceMapping=buildInstanceMapping(args.tagged_entities,args.auto_ceiling??true);
    applyPass(scene,activePass,instanceMapping.idToInstance,args.unit_scale??1.0,args.proxy?.bounds);
  }
  const project=(p)=>({x:((p[0]-origin.x-center.x)/(viewHeight*width/height)+.5)*width,
    y:(.5-(p[1]-origin.y-center.y)/viewHeight)*height});
  const world=(x,y)=>[origin.x+center.x+(x/width-.5)*viewHeight*width/height,
    origin.y+center.y+(.5-y/height)*viewHeight];
  const heading=(position,point)=>((Math.atan2(point[1]-position[1],point[0]-position[0])*180/Math.PI)%360+360)%360;
  const cameraSnapshot=()=>({cameras:clone(cameras),active_camera:active});
  function sendSnapshot(snapshot) {
    if(failed)return;
    pending=crypto.randomUUID();
    get('camera-save').textContent='Saving…';
    send('streamlit:setComponentValue',{dataType:'json',value:{type:'cameras',event_id:pending,
      expected_revision:revision,...snapshot}});
  }
  function commit() {
    dirty=false;
    const snapshot=cameraSnapshot();
    if(pending) {queued=snapshot;get('camera-save').textContent='Saving…';}
    else sendSnapshot(snapshot);
  }
  function ensureCameraTarget(data) {
    if(!data) return;
    if(!data.output) data.output = {width: 1024, height: 1024};
    if(!data.target || !Array.isArray(data.target) || data.target.length !== 3) {
      const h = (data.heading_deg ?? 0) * Math.PI / 180, p = (data.pitch_deg ?? 0) * Math.PI / 180;
      const d = 5000 * (args.unit_scale ?? 1);
      data.target = [
        data.position[0] + d * Math.cos(p) * Math.cos(h),
        data.position[1] + d * Math.cos(p) * Math.sin(h),
        data.position[2] + d * Math.sin(p)
      ];
    }
  }
  function syncControls() {
    const options=get('active-camera');options.replaceChildren();
    if(!Object.keys(cameras).length)options.add(new Option('No cameras yet',''));
    for(const [id,data] of Object.entries(cameras))options.add(new Option(data.name,id));
    options.value=active??'';
    const data=cameras[active];
    if(data) ensureCameraTarget(data);
    for(const [id,value] of [
      ['camera-name',data?.name??''],['camera-height',data?.position[2]??''],
      ['camera-heading',data?.heading_deg??''],['camera-pitch',data?.pitch_deg??''],['camera-fov',data?.fov_deg??''],
      ['camera-out-w',data?.output?.width??1024],['camera-out-h',data?.output?.height??1024]
    ]) {
      const input=get(id);
      if(input) {
        input.disabled=!data||failed||!args.units_known;
        if(document.activeElement!==input)input.value=typeof value==='number'?Number(value.toFixed(6)):value;
      }
    }
    const presetSelect=get('camera-aspect-preset');
    if(presetSelect) {
      presetSelect.disabled=!data||failed||!args.units_known;
      if(data?.output) {
        const w=data.output.width, h=data.output.height;
        if(w===1024&&h===1024) presetSelect.value='1:1';
        else if(w===1024&&h===768) presetSelect.value='4:3';
        else if(w===1200&&h===800) presetSelect.value='3:2';
        else if(w===1376&&h===768) presetSelect.value='16:9';
        else if(w===1920&&h===1080) presetSelect.value='16:9-fhd';
        else presetSelect.value='custom';
      }
    }
    get('delete-camera').disabled=!data||failed||!args.units_known;
    get('active-camera').disabled=!data||failed||!args.units_known;
    get('add-camera').disabled=failed||!args.units_known;
    const exportBtn=get('export-outputs');
    if(exportBtn) exportBtn.disabled=!data||failed||!args.units_known||exporting;
    get('height-unit').textContent=`(${args.units})`;
    if(!failed)get('camera-message').textContent=args.units_known?'':'Confirm drawing units in Layout settings before editing cameras.';
  }
  function drawMarkers() {
    const svg=get('camera-markers');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const element=(tag,attributes,text)=>{
      const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
      for(const [k,v] of Object.entries(attributes))el.setAttribute(k,String(v));
      if(text)el.textContent=text;svg.append(el);return el;
    };
    for(const [id,data] of Object.entries(cameras)) {
      ensureCameraTarget(data);
      const p=project(data.position), tp=project(data.target);
      const angle=data.heading_deg*Math.PI/180, half=data.fov_deg*Math.PI/360;
      const color=id===active?'#65e0c4':'#899cb0';
      const coneDist=Math.max(45,Math.min(Math.hypot(tp.x-p.x,tp.y-p.y),130));
      const endpoint=a=>({x:p.x+coneDist*Math.cos(a),y:p.y-coneDist*Math.sin(a)});
      const a=endpoint(angle-half),b=endpoint(angle+half);

      // FOV cone
      element('path',{d:`M${a.x},${a.y} L${p.x},${p.y} L${b.x},${b.y} L${a.x},${a.y}`,
        fill:color,'fill-opacity':.08,stroke:color,'stroke-opacity':.55,'data-camera-id':id});

      // Connecting line: ● ───────────────→ ◎
      element('line',{x1:p.x,y1:p.y,x2:tp.x,y2:tp.y,stroke:color,'stroke-width':2,'stroke-dasharray':'5 3','data-camera-id':id});

      // Direction arrow pointing to target
      const arrowAngle=Math.atan2(p.y-tp.y,tp.x-p.x);
      const arrowA={x:tp.x-12*Math.cos(arrowAngle-.4),y:tp.y+12*Math.sin(arrowAngle-.4)};
      const arrowB={x:tp.x-12*Math.cos(arrowAngle+.4),y:tp.y+12*Math.sin(arrowAngle+.4)};
      element('path',{d:`M${arrowA.x},${arrowA.y} L${tp.x},${tp.y} L${arrowB.x},${arrowB.y}`,fill:'none',stroke:color,'stroke-width':2});

      // Camera position dot ●
      element('circle',{cx:p.x,cy:p.y,r:7,fill:color,stroke:'#111b25','stroke-width':2,'data-position-id':id,style:'cursor:grab'});

      // Target look-at handle ◎
      element('circle',{cx:tp.x,cy:tp.y,r:8,fill:'#111b25',stroke:color,'stroke-width':2,'data-target-id':id,style:'cursor:grab'});
      element('circle',{cx:tp.x,cy:tp.y,r:3,fill:color,'data-target-id':id,style:'cursor:grab'});

      element('text',{x:p.x+10,y:p.y-12,fill:color,'font-size':12},data.name);
    }
  }
  function render() {
    const isCamera = args.mode === 'camera';
    if(isCamera) {
      const half=viewHeight/2;
      planCamera.left=center.x-half*width/height;planCamera.right=center.x+half*width/height;
      planCamera.top=center.y+half;planCamera.bottom=center.y-half;planCamera.updateProjectionMatrix();
      scene.background.setHex(0x111b25);planRenderer.render(scene,planCamera);drawMarkers();
      get('plan-hint').textContent=adding?'Click and drag to place and aim · Esc: cancel':
        'Drag camera dot ●: move · Drag target ◎: aim · Empty space: pan · Scroll: zoom';
    }
    const data=cameras[active];
    get('perspective-empty').textContent=!data?'Place a camera on the plan to preview this Layout.':
      (!args.proxy?.meshes?.length?'No proxy geometry yet. Tag closed floor/furniture footprints or walls in Annotation Mode.':'');
    if(data) {
      ensureCameraTarget(data);
      const out=data.output||{width:1024,height:1024};
      const aspect=out.width/out.height;

      // Letterbox / Pillarbox preview sizing without distortion
      const hostW=previewHost.clientWidth||previewWidth;
      const hostH=previewHost.clientHeight||previewHeight;
      let drawW,drawH;
      if(hostW/hostH>aspect) {
        drawH=hostH;
        drawW=Math.round(hostH*aspect);
      } else {
        drawW=hostW;
        drawH=Math.round(hostW/aspect);
      }
      previewRenderer.setSize(drawW,drawH);

      if(activePass==='planar') {
        const floorElev=args.floor_elevation??0.0;
        const planarCanvas=renderPlanarCanvas(geometry,args.tags,args.entity_colors,data,floorElev,out.width,out.height);
        const texture=new THREE.CanvasTexture(planarCanvas);
        if(planarQuad.material.map) planarQuad.material.map.dispose();
        planarQuad.material.map=texture;
        planarQuad.material.needsUpdate=true;
        previewRenderer.render(planarScene,planarOrthoCamera);
      } else {
        configurePerspective(perspective,data,aspect,origin,args.proxy?.bounds,args.unit_scale??1);
        const bgColor = activePass === 'depth' ? 0x000000 : (activePass === 'proxy' ? 0x29333d : 0x111b25);
        scene.background.setHex(bgColor);
        previewRenderer.render(scene,perspective);
      }
      get('perspective-status').textContent=`${data.name} · [${activePass.toUpperCase()}] · ${out.width}×${out.height} (${Number(aspect.toFixed(2))}:1) · H ${Number(data.position[2].toFixed(3))} ${args.units} · FOV ${Number(data.fov_deg.toFixed(1))}° · ${args.proxy?.meshes?.length??0} proxy parts`;
    } else {previewRenderer.setClearColor(0x29333d);previewRenderer.clear();get('perspective-status').textContent='No active camera';}
    // Exposed immutable diagnostics for scientific/browser checks, not an alternate state store.
    root.dataset.activeCamera=active??'';
    root.dataset.cameraState=JSON.stringify(data??null);
    root.dataset.projection=JSON.stringify(data?[...perspective.projectionMatrix.elements,previewWidth,previewHeight]:[]);
    root.dataset.view=JSON.stringify({origin,center,viewHeight,width,height});
  }
  function fit() {
    const b=boundsOf(geometry),xs=[b.xmin,b.xmax],ys=[b.ymin,b.ymax];
    for(const data of Object.values(cameras)){xs.push(data.position[0]);ys.push(data.position[1]);}
    center={x:(Math.min(...xs)+Math.max(...xs))/2-origin.x,y:(Math.min(...ys)+Math.max(...ys))/2-origin.y};
    viewHeight=Math.max(Math.max(...ys)-Math.min(...ys),(Math.max(...xs)-Math.min(...xs))*height/width,1)*1.25;render();
  }
  function resize() {
    const isCamera = args.mode === 'camera';
    const activeWorkspace = isCamera ? root : document.getElementById('annotation-workspace');
    if(!activeWorkspace) return;
    previewWidth=previewHost.clientWidth;previewHeight=previewHost.clientHeight;
    if(isCamera) {
      width=planHost.clientWidth;height=planHost.clientHeight;
      if(width&&height) planRenderer.setSize(width,height);
    }
    render();
    send('streamlit:setFrameHeight',{height:Math.max(activeWorkspace.offsetHeight+4,640)});
  }
  new ResizeObserver(resize).observe(root);
  const annotationWorkspace=document.getElementById('annotation-workspace');
  if(annotationWorkspace) new ResizeObserver(resize).observe(annotationWorkspace);
  get('camera-fit').onclick=fit;
  get('add-camera').onclick=()=>{adding=true;get('add-camera').textContent='Click + drag on plan…';render();};
  get('delete-camera').onclick=()=>{
    if(!active)return;delete cameras[active];active=Object.keys(cameras)[0]??null;adding=false;syncControls();render();commit();
  };
  get('active-camera').onchange=event=>{active=event.target.value;adding=false;syncControls();render();commit();};
  const passButtons=root.querySelectorAll('.pass-tab');
  passButtons.forEach(btn=>{
    btn.onclick=()=>{
      activePass=btn.dataset.pass;
      passButtons.forEach(b=>{
        const isActive=b===btn;
        b.classList.toggle('active',isActive);
        b.setAttribute('aria-selected',isActive?'true':'false');
      });
      updateActivePass();
      render();
    };
  });
  const exportBtn=get('export-outputs');
  if(exportBtn) {
    exportBtn.onclick=async ()=>{
      const data=cameras[active];
      if(!data||exporting)return;
      exporting=true;
      exportBtn.disabled=true;
      const originalText=exportBtn.textContent;
      exportBtn.textContent='Exporting…';
      get('perspective-status').textContent='Rendering conditioning passes…';
      try {
        ensureCameraTarget(data);
        const out=data.output||{width:1024,height:1024};
        const w=parseInt(out.width,10), h=parseInt(out.height,10);
        const exportCam=perspective.clone();
        exportCam.layers.set(1);
        const aspect=w/h;
        configurePerspective(exportCam,data,aspect,origin,args.proxy?.bounds,args.unit_scale??1);
        const instanceMapping=buildInstanceMapping(args.tagged_entities,args.auto_ceiling??true);
        const passes=renderOffscreenPasses(
          scene,
          exportCam,
          w,
          h,
          instanceMapping.idToInstance,
          args.unit_scale??1.0,
          args.proxy?.bounds,
          {
            geometry,
            tags: args.tags,
            entityColors: args.entity_colors,
            cameraData: data,
            floorElevation: args.floor_elevation ?? 0.0
          }
        );
        updateActivePass();
        render();
        const eventId=crypto.randomUUID();
        send('streamlit:setComponentValue',{
          dataType:'json',
          value:{
            type:'export_conditioning',
            event_id:eventId,
            camera_id:active,
            resolution:{width:w,height:h},
            ...passes
          }
        });
        get('perspective-status').textContent=`Exported conditioning passes for ${data.name} (${w}×${h})`;
      } catch(err) {
        console.error('Export conditioning error:',err);
        get('perspective-status').textContent='Export error: '+err.message;
      } finally {
        exporting=false;
        exportBtn.disabled=false;
        exportBtn.textContent=originalText;
      }
    };
  }
  for(const [id,key] of [
    ['camera-name','name'],['camera-height','height'],['camera-heading','heading_deg'],
    ['camera-pitch','pitch_deg'],['camera-fov','fov_deg'],['camera-out-w','out_w'],['camera-out-h','out_h']
  ]) {
    const input=get(id);
    if(!input) continue;
    input.oninput=()=>{
      const data=cameras[active];if(!data)return;
      ensureCameraTarget(data);
      const value=key==='name'?input.value:Number(input.value);
      const valid=key==='name'?!!value.trim():input.value!==''&&Number.isFinite(value)&&
        (key!=='pitch_deg'||Math.abs(value)<90)&&(key!=='fov_deg'||value>0&&value<180)&&
        ((key!=='out_w'&&key!=='out_h')||(value>=16&&value<=16384));
      input.setCustomValidity(valid?'':'Enter a valid camera value');
      if(!valid)return;
      if(key==='height') {
        const diff=value-data.position[2];
        data.position[2]=value;
        data.target[2]+=diff;
      } else if(key==='heading_deg'||key==='pitch_deg') {
        data[key]=key==='heading_deg'?((value%360)+360)%360:value;
        const d=Math.hypot(data.target[0]-data.position[0],data.target[1]-data.position[1],data.target[2]-data.position[2])||(5000*(args.unit_scale??1));
        const h=data.heading_deg*Math.PI/180, p=data.pitch_deg*Math.PI/180;
        data.target=[data.position[0]+d*Math.cos(p)*Math.cos(h),data.position[1]+d*Math.cos(p)*Math.sin(h),data.position[2]+d*Math.sin(p)];
      } else if(key==='out_w') {
        data.output.width=Math.round(value);
      } else if(key==='out_h') {
        data.output.height=Math.round(value);
      } else {
        data[key]=value;
      }
      dirty=true;get('camera-save').textContent='Editing…';render();
    };
    input.onchange=()=>{if(input.checkValidity()){syncControls();commit();}else input.reportValidity();};
  }
  const presetSelect=get('camera-aspect-preset');
  if(presetSelect) {
    presetSelect.onchange=()=>{
      const data=cameras[active];if(!data)return;
      ensureCameraTarget(data);
      if(presetSelect.value==='1:1'){data.output.width=1024;data.output.height=1024;}
      else if(presetSelect.value==='4:3'){data.output.width=1024;data.output.height=768;}
      else if(presetSelect.value==='3:2'){data.output.width=1200;data.output.height=800;}
      else if(presetSelect.value==='16:9'){data.output.width=1376;data.output.height=768;}
      else if(presetSelect.value==='16:9-fhd'){data.output.width=1920;data.output.height=1080;}
      dirty=true;syncControls();render();commit();
    };
  }
  const local=e=>{const box=canvas.getBoundingClientRect();return{x:e.clientX-box.left,y:e.clientY-box.top};};
  canvas.oncontextmenu=e=>e.preventDefault();
  canvas.addEventListener('pointerdown',e=>{
    if(failed||![0,1,2].includes(e.button))return;
    const p=local(e),point=world(p.x,p.y);let action='pan',id=null;
    const before=cameraSnapshot();
    if(e.button===0&&adding&&args.units_known) {
      let n=1;while(cameras[`camera_${String(n).padStart(3,'0')}`])n++;
      id=`camera_${String(n).padStart(3,'0')}`;
      const d=5000*(args.unit_scale??1);
      cameras[id]={name:`Camera ${String(n).padStart(3,'0')}`,position:[...point,1500*(args.unit_scale??1)],
        target:[point[0]+d,point[1],1500*(args.unit_scale??1)],heading_deg:0,pitch_deg:0,fov_deg:60,output:{width:1024,height:1024}};
      active=id;action='target';
    } else if(e.button===0&&args.units_known) {
      for(const [candidate,data] of Object.entries(cameras).sort(([a],[b])=>a===active?-1:b===active?1:0)) {
        ensureCameraTarget(data);
        const cp=project(data.position), tp=project(data.target);
        if(Math.hypot(p.x-tp.x,p.y-tp.y)<14){id=candidate;action='target';break;}
        if(Math.hypot(p.x-cp.x,p.y-cp.y)<14){id=candidate;action='move';break;}
        const h=data.heading_deg*Math.PI/180;
        if(Math.hypot(p.x-cp.x-85*Math.cos(h),p.y-cp.y+85*Math.sin(h))<12){id=candidate;action='target';break;}
      }
      if(id)active=id;
    }
    down={...p,action,id,before,cx:center.x,cy:center.y,start:point,
      position:id?[...cameras[id].position]:null,target:id?[...cameras[id].target]:null};
    canvas.setPointerCapture(e.pointerId);syncControls();render();
  });
  canvas.addEventListener('pointermove',e=>{
    if(!down)return;
    const p=local(e),point=world(p.x,p.y);
    if(down.action==='pan')center={x:down.cx-(p.x-down.x)*viewHeight/height,y:down.cy+(p.y-down.y)*viewHeight/height};
    else if(down.action==='target') {
      const data=cameras[down.id];
      ensureCameraTarget(data);
      data.target[0]=point[0];
      data.target[1]=point[1];
      const dx=data.target[0]-data.position[0];
      const dy=data.target[1]-data.position[1];
      const dz=data.target[2]-data.position[2];
      const dxy=Math.hypot(dx,dy);
      if(dxy>1e-6) {
        data.heading_deg=((Math.atan2(dy,dx)*180/Math.PI)%360+360)%360;
        data.pitch_deg=Math.max(-89,Math.min(89,Math.atan2(dz,dxy)*180/Math.PI));
      }
      get('camera-save').textContent='Editing…';syncControls();
    } else if(down.action==='move') {
      const data=cameras[down.id];
      ensureCameraTarget(data);
      const dx=point[0]-down.start[0];
      const dy=point[1]-down.start[1];
      data.position=[down.position[0]+dx,down.position[1]+dy,data.position[2]];
      if(down.target) {
        data.target=[down.target[0]+dx,down.target[1]+dy,data.target[2]];
      }
      get('camera-save').textContent='Editing…';syncControls();
    }
    render();
  });
  canvas.addEventListener('pointerup',e=>{
    if(!down)return;const changed=down.action!=='pan';down=null;
    canvas.releasePointerCapture(e.pointerId);
    if(changed){adding=false;get('add-camera').textContent='+ Add Camera';syncControls();render();commit();}
  });
  function cancel() {
    if(down){cameras=down.before.cameras;active=down.before.active_camera;down=null;}
    adding=false;get('add-camera').textContent='+ Add Camera';syncControls();render();
    get('camera-save').textContent=pending?'Saving…':'Saved';
  }
  canvas.addEventListener('pointercancel',cancel);
  window.addEventListener('keydown',e=>{if(e.key==='Escape')cancel();});
  canvas.addEventListener('wheel',e=>{
    e.preventDefault();const p=local(e),before=world(p.x,p.y);
    viewHeight=Math.max(1e-6,Math.min(1e15,viewHeight*Math.exp(Math.max(-1,Math.min(1,e.deltaY*.001)))));
    const after=world(p.x,p.y);center.x+=before[0]-after[0];center.y+=before[1]-after[1];render();
  },{passive:false});
  function update(next) {
    args=next;
    const isCamera=next.mode==='camera';
    root.hidden=!isCamera;
    const annotationWorkspace=document.getElementById('annotation-workspace');
    if(annotationWorkspace) annotationWorkspace.hidden=isCamera;

    const annotationHost=document.getElementById('annotation-preview-host');
    const cameraPreviewPane=root.querySelector('.preview-pane');

    if(isCamera) {
      if(previewHost.parentElement!==cameraPreviewPane) cameraPreviewPane.append(previewHost);
    } else {
      if(annotationHost&&previewHost.parentElement!==annotationHost) annotationHost.append(previewHost);
    }

    revision=args.revision;
    if(args.camera_error){failed=true;pending=null;queued=null;get('camera-message').textContent=args.camera_error+' Reopen the Layout to reload saved camera state.';get('camera-save').textContent='Not saved';}
    if(pending&&args.acknowledged_event===pending) {
      pending=null;
      if(queued&&!failed){const snapshot=queued;queued=null;sendSnapshot(snapshot);}
      else if(!failed)get('camera-save').textContent='Saved';
    }
    if(!pending&&!queued&&!down&&!failed&&!dirty){
      cameras=clone(args.cameras??{});
      for(const cam of Object.values(cameras)) ensureCameraTarget(cam);
      active=args.active_camera;
    }
    let refit=false;
    if(geometryVersion!==args.geometry_version) {
      geometry=args.entities??[];geometryVersion=args.geometry_version;
      const b=boundsOf(geometry);origin={x:(b.xmin+b.xmax)/2,y:(b.ymin+b.ymax)/2};
      if(lines)disposeGroup(lines);lines=buildLines(geometry,origin);scene.add(lines);refit=true;
    }
    for(const line of lines?.children??[]) {
      const id = line.userData.id;
      const col = args.entity_colors?.[id] != null
        ? (typeof args.entity_colors[id] === 'string' ? parseInt(args.entity_colors[id].replace('#',''), 16) : args.entity_colors[id])
        : (colors[args.tags?.[id]] ?? colors.untagged);
      line.material.color.setHex(col);
    }
    if(proxyVersion!==args.proxy_version||refit){
      if(proxy)disposeGroup(proxy);
      proxy=buildProxy(args.proxy?.meshes??[],origin);
      scene.add(proxy);
      proxyVersion=args.proxy_version;
      updateActivePass();
    }
    if(isCamera) {
      updateActivePass();
      syncControls();
    }
    resize();
    if(refit&&isCamera) fit(); else render();
  }
  return {update,root,scene,perspective};
}
