import * as THREE from './vendor/three.module.js';
export const colors = {wall:0xfbad65, floor:0x65cbb7, furniture:0xb99af7, void:0x6bbfff, untagged:0xa5b8c8};

export function boundsOf(entities) {
  let xmin=Infinity,ymin=Infinity,xmax=-Infinity,ymax=-Infinity;
  for (const e of entities) for (const path of e.paths) for (const [x,y] of path) {
    xmin=Math.min(xmin,x);ymin=Math.min(ymin,y);xmax=Math.max(xmax,x);ymax=Math.max(ymax,y);
  }
  return Number.isFinite(xmin) ? {xmin,ymin,xmax,ymax} : {xmin:0,ymin:0,xmax:100,ymax:100};
}

export function buildLines(entities, origin) {
  const group = new THREE.Group();
  for (const entity of entities) {
    const positions=[];
    for (const path of entity.paths) for(let i=1;i<path.length;i++) {
      for (const p of [path[i-1],path[i]]) positions.push(p[0]-origin.x,p[1]-origin.y,0);
    }
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    const line=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:colors.untagged}));
    line.userData.id=entity.id;
    group.add(line);
  }
  return group;
}
