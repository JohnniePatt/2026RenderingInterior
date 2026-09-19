import * as THREE from './vendor/three.module.js';
import {boundsOf, buildLines, colors} from './geometry.js';
import {pickEntity} from './selection.js';
import {createCameraWorkspace} from './camera_workspace.js';

const host=document.getElementById('viewport'), error=document.getElementById('error');
const send=(type, data={})=>window.parent.postMessage({isStreamlitMessage:true,type,...data},'*');
let renderer;
let cameraWorkspace;
try { renderer=new THREE.WebGLRenderer({antialias:true}); }
catch(e) { error.textContent='WebGL is unavailable. Enable browser hardware acceleration. You can also select entities in the Properties panel.'; }
if (renderer) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.setClearColor(0x111b25);
  host.prepend(renderer.domElement);
  const canvas=renderer.domElement;
  canvas.setAttribute('aria-label','Interactive CAD plan');
  canvas.tabIndex=0;
  const scene=new THREE.Scene(), camera=new THREE.OrthographicCamera(-1,1,1,-1,0.1,100);
  camera.position.z=10;
  let args={}, entities=[], selected=new Set(), lines=null, version=null, origin={x:0,y:0}, pendingEvent=null;
  let center={x:0,y:0}, viewHeight=100, width=1, height=610, down=null;
  const project=(p)=>({x:((p[0]-origin.x-center.x)/(viewHeight*width/height)+0.5)*width,
                       y:(0.5-(p[1]-origin.y-center.y)/viewHeight)*height});
  const world=(x,y)=>({x:center.x+(x/width-0.5)*viewHeight*width/height,
                       y:center.y+(0.5-y/height)*viewHeight});
  function draw() {
    const half=viewHeight/2, ratio=width/height;
    camera.left=center.x-half*ratio;camera.right=center.x+half*ratio;
    camera.top=center.y+half;camera.bottom=center.y-half;camera.updateProjectionMatrix();
    if(lines) for(const line of lines.children) {
      const id=line.userData.id;
      const col = args.entity_colors?.[id] != null
        ? (typeof args.entity_colors[id] === 'string' ? parseInt(args.entity_colors[id].replace('#',''), 16) : args.entity_colors[id])
        : (colors[args.tags?.[id]] ?? colors.untagged);
      line.material.color.setHex(selected.has(id)?0xffffff:col);
    }
    renderer.render(scene,camera);
    document.getElementById('status').textContent=`${entities.length} entities · ${selected.size} selected · ${args.units??'unknown'} · ${Math.round(viewHeight)} units high`;
  }
  function fit() {
    const b=boundsOf(entities);
    center={x:(b.xmin+b.xmax)/2-origin.x,y:(b.ymin+b.ymax)/2-origin.y};
    viewHeight=Math.max(b.ymax-b.ymin,(b.xmax-b.xmin)*height/width,1)*1.25;
    draw();
  }
  function emit() {
    pendingEvent=crypto.randomUUID();
    send('streamlit:setComponentValue',{value:{event_id:pendingEvent,selected_ids:[...selected]},dataType:'json'});
  }
  function resize() { width=host.clientWidth;height=host.clientHeight;renderer.setSize(width,height);draw(); }
  new ResizeObserver(resize).observe(host);
  document.getElementById('fit').onclick=fit;
  document.getElementById('clear').onclick=()=>{selected.clear();draw();emit();};
  canvas.oncontextmenu=e=>e.preventDefault();
  canvas.addEventListener('pointerdown',e=>{
    if(![0,1,2].includes(e.button)) return;
    down={x:e.offsetX,y:e.offsetY,cx:center.x,cy:center.y,button:e.button,moved:false};
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    if(!down)return;
    const dx=e.offsetX-down.x,dy=e.offsetY-down.y;
    if(Math.hypot(dx,dy)>3) down.moved=true;
    if(down.moved){center={x:down.cx-dx*viewHeight/height,y:down.cy+dy*viewHeight/height};draw();}
  });
  canvas.addEventListener('pointerup',e=>{
    if(!down)return;
    if(!down.moved && down.button===0){
      const id=pickEntity({x:e.offsetX,y:e.offsetY},entities,project);
      if(!e.shiftKey&&!e.ctrlKey&&!e.metaKey)selected.clear();
      if(id){if(selected.has(id))selected.delete(id);else selected.add(id);}
      draw();emit();
    }
    down=null;
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointercancel',()=>{down=null;});
  canvas.addEventListener('wheel',e=>{
    e.preventDefault();const before=world(e.offsetX,e.offsetY);
    viewHeight=Math.max(1e-6,Math.min(1e15,viewHeight*Math.exp(Math.max(-1,Math.min(1,e.deltaY*0.001)))));
    const after=world(e.offsetX,e.offsetY);center.x+=before.x-after.x;center.y+=before.y-after.y;draw();
  },{passive:false});
  window.addEventListener('message',event=>{
    if(event.source!==window.parent||event.data.type!=='streamlit:render')return;
    args=event.data.args;
    host.hidden=args.mode==='camera';
    try {cameraWorkspace??=createCameraWorkspace(send);window.__cameraWorkspace=cameraWorkspace;cameraWorkspace.update(args);}
    catch(err){host.hidden=false;error.textContent=`Camera preview error: ${err.message}`;}
    if(args.mode==='camera')return;
    // Ignore obsolete Python renders while a newer local click is in flight.
    if(!pendingEvent || args.acknowledged_event===pendingEvent) {
      selected=new Set(args.selected_ids??[]);
      pendingEvent=null;
    }
    if(version!==args.geometry_version){
      entities=args.entities??[];version=args.geometry_version;
      if(lines){scene.remove(lines);for(const line of lines.children){line.geometry.dispose();line.material.dispose();}}
      const b=boundsOf(entities);origin={x:(b.xmin+b.xmax)/2,y:(b.ymin+b.ymax)/2};
      lines=buildLines(entities,origin);scene.add(lines);resize();fit();
    }
    draw();send('streamlit:setFrameHeight',{height:640});
  });
}
send('streamlit:componentReady',{apiVersion:1});
send('streamlit:setFrameHeight',{height:640});
