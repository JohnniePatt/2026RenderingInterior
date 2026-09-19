import * as THREE from './vendor/three.module.js';
import { colors as semanticColors } from './geometry.js';

export function disposeGroup(group) {
  group.traverse(object=>{object.geometry?.dispose();if(object.material) {
    for(const material of (Array.isArray(object.material)?object.material:[object.material])) material.dispose();
  }});
  group.removeFromParent();
}

export function buildProxy(descriptors, origin) {
  const group=new THREE.Group();
  const colors = {
    wall: semanticColors?.wall ?? 0xfbad65,
    floor: semanticColors?.floor ?? 0x65cbb7,
    furniture: semanticColors?.furniture ?? 0xb99af7,
    void: semanticColors?.void ?? 0x6bbfff,
    ceiling: 0xdee5ed
  };
  for(const item of descriptors) {
    let geometry;
    if(item.kind==='solid') {
      const vectors=points=>points.map(p=>new THREE.Vector2(p[0]-origin.x,p[1]-origin.y));
      const shape=new THREE.Shape(vectors(item.outer));
      shape.holes=item.holes.map(ring=>new THREE.Path(vectors(ring)));
      const depth=item.z_max-item.z_min;
      geometry=depth>0 ? new THREE.ExtrudeGeometry(shape,{depth,bevelEnabled:false,steps:1}) : new THREE.ShapeGeometry(shape);
      geometry.translate(0,0,item.z_min);
    } else {
      const positions=[];
      for(let i=1;i<item.path.length;i++) {
        const a=item.path[i-1],b=item.path[i];
        const points=[[a[0],a[1],item.z_min],[b[0],b[1],item.z_min],
          [b[0],b[1],item.z_max],[a[0],a[1],item.z_max]];
        for(const j of [0,1,2,0,2,3]) positions.push(points[j][0]-origin.x,points[j][1]-origin.y,points[j][2]);
      }
      geometry=new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
      geometry.computeVertexNormals();
    }
    const baseColor = item.color != null
      ? (typeof item.color === 'string' ? parseInt(item.color.replace('#',''), 16) : item.color)
      : (colors[item.semantic] ?? 0xadb9c4);
    const matOptions = {
      color: baseColor,
      roughness: 0.82,
      metalness: 0.05,
      side: THREE.DoubleSide
    };
    if (item.semantic === 'void') {
      matOptions.color = colors.void;
      matOptions.transparent = true;
      matOptions.opacity = 0.55;
      matOptions.roughness = 0.2;
    } else if (item.semantic === 'ceiling') {
      matOptions.color = 0xf8fafc;
      matOptions.emissive = 0x55606e;
      matOptions.roughness = 0.95;
    } else if (item.semantic === 'furniture') {
      matOptions.roughness = 0.55;
      matOptions.transparent = true;
      matOptions.opacity = 0.88;
    }
    const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial(matOptions));
    mesh.layers.set(1);
    mesh.userData={entityId:item.entity_id, semantic:item.semantic, dimensionSource:item.dimension_source};
    group.add(mesh);
    const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,25),new THREE.LineBasicMaterial({color:0x1b2836,transparent:true,opacity:.5}));
    edges.layers.set(1);group.add(edges);
  }
  return group;
}
