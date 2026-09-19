import * as THREE from './vendor/three.module.js';

export function configurePerspective(camera, data, aspect, origin, bounds, unitScale=1) {
  const h=THREE.MathUtils.degToRad(data.heading_deg), p=THREE.MathUtils.degToRad(data.pitch_deg);
  camera.position.set(data.position[0]-origin.x,data.position[1]-origin.y,data.position[2]);
  camera.up.set(0,0,1); // Keep DXF Z up throughout; no Y/Z swap or metric scaling.
  const forward=new THREE.Vector3(Math.cos(p)*Math.cos(h),Math.cos(p)*Math.sin(h),Math.sin(p));
  camera.lookAt(camera.position.clone().add(forward));
  camera.aspect=aspect;
  camera.fov=THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(data.fov_deg)/2)/aspect));
  const corners=bounds ? [new THREE.Vector3(bounds[0][0]-origin.x,bounds[0][1]-origin.y,bounds[0][2]),
    new THREE.Vector3(bounds[1][0]-origin.x,bounds[1][1]-origin.y,bounds[1][2])] : [];
  const diagonal=corners.length ? corners[0].distanceTo(corners[1]) : 10000*unitScale;
  const distance=corners.length ? camera.position.distanceTo(corners[0].clone().add(corners[1]).multiplyScalar(.5)) : 0;
  camera.near=Math.max(50*unitScale,Math.min(200*unitScale,diagonal*1e-3));
  camera.far=Math.max(camera.near*100, distance+diagonal*2,1000*unitScale);
  camera.updateProjectionMatrix();camera.updateMatrixWorld();
}
