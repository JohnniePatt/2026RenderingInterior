import * as THREE from './vendor/three.module.js';
import { renderPlanarCanvas } from './homography.js';

export const SEMANTIC_CLASSES = {
  background: 0,
  wall: 1,
  floor: 2,
  furniture: 3,
  void: 4,
  ceiling: 5
};

export const SEMANTIC_COLORS = {
  0: 0x111b25, // Background
  1: 0xfbad65, // Wall
  2: 0x65cbb7, // Floor
  3: 0xb99af7, // Furniture
  4: 0x6bbfff, // Void (Door / Window)
  5: 0xdee5ed  // Ceiling
};

export const INSTANCE_PALETTE = [
  0xe6194b, 0x3cb44b, 0xffe119, 0x4363d8, 0xf58231,
  0x911eb4, 0x42d4f4, 0xf032e6, 0xbfef45, 0xfabebe,
  0x469990, 0xe6beff, 0x9a6324, 0xfffac8, 0x800000,
  0xaaffc3, 0x808000, 0xffd8b1, 0x000075, 0x808080
];

export function buildInstanceMapping(entities, autoCeiling = true) {
  const sortedIds = Object.keys(entities || {})
    .filter(id => entities[id] && !entities[id].orphaned && entities[id].semantic)
    .sort();
  const mapping = { 0: 'background' };
  sortedIds.forEach((id, idx) => {
    mapping[idx + 1] = id;
  });
  if (autoCeiling) {
    mapping[Object.keys(mapping).length] = 'auto_ceiling';
  }
  const idToInstance = {};
  for (const [instId, entId] of Object.entries(mapping)) {
    idToInstance[entId] = parseInt(instId, 10);
  }
  return { mapping, idToInstance };
}

// Shaders for raw and visual depth
const depthRawVertexShader = `
  varying float vDepthMm;
  uniform float uToMm;
  void main() {
    vec4 viewPos = modelViewMatrix * vec4(position, 1.0);
    // viewPos.z is negative forward in Three.js camera space
    float forwardDepth = -viewPos.z;
    vDepthMm = max(0.0, forwardDepth * uToMm);
    gl_Position = projectionMatrix * viewPos;
  }
`;

const depthRawFragmentShader = `
  varying float vDepthMm;
  void main() {
    float d = floor(vDepthMm + 0.5);
    float r = mod(d, 256.0);
    float g = mod(floor(d / 256.0), 256.0);
    float b = mod(floor(d / 65536.0), 256.0);
    gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);
  }
`;

const depthVisVertexShader = `
  varying float vDepth;
  uniform float uToM;
  void main() {
    vec4 viewPos = modelViewMatrix * vec4(position, 1.0);
    vDepth = -viewPos.z * uToM;
    gl_Position = projectionMatrix * viewPos;
  }
`;

const depthVisFragmentShader = `
  varying float vDepth;
  uniform float uNear;
  uniform float uFar;
  void main() {
    float norm = clamp((vDepth - uNear) / max(0.001, (uFar - uNear)), 0.0, 1.0);
    float val = 1.0 - norm; // Closer is brighter
    gl_FragColor = vec4(val, val, val, 1.0);
  }
`;

export function createDepthRawMaterial(unitScale = 1.0) {
  const toMm = unitScale > 0 ? (1.0 / unitScale) : 1000.0;
  return new THREE.ShaderMaterial({
    vertexShader: depthRawVertexShader,
    fragmentShader: depthRawFragmentShader,
    uniforms: { uToMm: { value: toMm } },
    side: THREE.DoubleSide
  });
}

export function createDepthVisMaterial(unitScale = 1.0, near = 0.5, far = 15.0) {
  const toM = unitScale > 0 ? (1.0 / (unitScale * 1000.0)) : 1.0;
  return new THREE.ShaderMaterial({
    vertexShader: depthVisVertexShader,
    fragmentShader: depthVisFragmentShader,
    uniforms: {
      uToM: { value: toM },
      uNear: { value: near },
      uFar: { value: far }
    },
    side: THREE.DoubleSide
  });
}

export function applyPass(scene, mode, idToInstance, unitScale = 1.0, bounds = null) {
  const toM = unitScale > 0 ? (1.0 / (unitScale * 1000.0)) : 1.0;
  let near = 0.2, far = 15.0;
  if (bounds && Array.isArray(bounds) && bounds.length === 2) {
    const diagM = Math.hypot(
      (bounds[1][0] - bounds[0][0]) * toM,
      (bounds[1][1] - bounds[0][1]) * toM,
      (bounds[1][2] - bounds[0][2]) * toM
    );
    far = Math.max(8.0, diagM * 1.5);
  }

  scene.traverse(obj => {
    if (obj.isLineSegments || obj.isLine) {
      if (obj.layers.isEnabled(1)) {
        obj.visible = (mode === 'proxy');
      }
    }
    if (obj.isMesh && obj.layers.isEnabled(1)) {
      if (!obj.userData.proxyMaterial) {
        obj.userData.proxyMaterial = obj.material;
      }
      const entId = obj.userData.entityId;
      const semantic = obj.userData.semantic;

      if (mode === 'proxy') {
        obj.material = obj.userData.proxyMaterial;
      } else if (mode === 'depth') {
        obj.material = createDepthVisMaterial(unitScale, near, far);
      } else if (mode === 'instance') {
        const instId = idToInstance[entId] ?? 0;
        const colorHex = instId > 0 ? INSTANCE_PALETTE[(instId - 1) % INSTANCE_PALETTE.length] : 0x000000;
        obj.material = new THREE.MeshBasicMaterial({ color: colorHex, side: THREE.DoubleSide });
      } else if (mode === 'semantic') {
        const classId = SEMANTIC_CLASSES[semantic] ?? 0;
        const colorHex = SEMANTIC_COLORS[classId] ?? 0x111b25;
        obj.material = new THREE.MeshBasicMaterial({ color: colorHex, side: THREE.DoubleSide });
      }
    }
  });

  if (mode === 'depth') {
    scene.background = new THREE.Color(0x000000);
  } else if (mode === 'instance' || mode === 'semantic') {
    scene.background = new THREE.Color(0x111b25);
  } else {
    scene.background = new THREE.Color(0x29333d);
  }
}

export function renderOffscreenPasses(scene, camera, width, height, idToInstance, unitScale = 1.0, bounds = null, planarContext = null) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: true,
    preserveDrawingBuffer: true
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(1);

  // Compute depth range from bounds if available
  const toM = unitScale > 0 ? (1.0 / (unitScale * 1000.0)) : 1.0;
  let near = 0.5, far = 15.0;
  if (bounds && Array.isArray(bounds) && bounds.length === 2) {
    const diagM = Math.hypot(
      (bounds[1][0] - bounds[0][0]) * toM,
      (bounds[1][1] - bounds[0][1]) * toM,
      (bounds[1][2] - bounds[0][2]) * toM
    );
    far = Math.max(8.0, diagM * 1.5);
  }

  // Helper to freeze materials and render
  const renderToDataUrl = (prepareFn, clearColor = 0x000000, clearAlpha = 1.0, showEdges = false, isRaw = false) => {
    renderer.outputColorSpace = isRaw ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace;
    scene.traverse(obj => {
      if (obj.isLineSegments || obj.isLine) {
        if (obj.layers.isEnabled(1)) obj.visible = showEdges;
      }
      if (obj.isMesh && obj.layers.isEnabled(1)) {
        if (!obj.userData.proxyMaterial) obj.userData.proxyMaterial = obj.material;
        obj.material = prepareFn(obj);
      }
    });
    renderer.setClearColor(clearColor, clearAlpha);
    renderer.clear();
    renderer.render(scene, camera);
    return canvas.toDataURL('image/png');
  };

  // 1. Proxy RGB
  const proxy_png = renderToDataUrl(obj => obj.userData.proxyMaterial, 0x29333d, 1.0, true, false);

  // 2. Depth Vis (Normalized grayscale)
  const depthVisMat = createDepthVisMaterial(unitScale, near, far);
  const depth_vis_png = renderToDataUrl(() => depthVisMat, 0x000000, 1.0, false, false);

  // 3. Depth Raw (Encoded mm in RGB, Alpha=0 for background)
  const depthRawMat = createDepthRawMaterial(unitScale);
  const depth_raw_png = renderToDataUrl(() => depthRawMat, 0x000000, 0.0, false, true);

  // 4. Instance Vis (Distinct color per instance, background [0, 0, 0])
  const instance_vis_png = renderToDataUrl(obj => {
    const instId = idToInstance[obj.userData.entityId] ?? 0;
    const col = instId > 0 ? INSTANCE_PALETTE[(instId - 1) % INSTANCE_PALETTE.length] : 0x000000;
    return new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide });
  }, 0x000000, 1.0, false, false);

  // 5. Instance Raw (R + G*256 encodes exact integer ID, Alpha=0 for background)
  const instance_raw_png = renderToDataUrl(obj => {
    const instId = idToInstance[obj.userData.entityId] ?? 0;
    const r = (instId % 256) / 255.0;
    const g = (Math.floor(instId / 256) % 256) / 255.0;
    return new THREE.MeshBasicMaterial({ color: new THREE.Color(r, g, 0), side: THREE.DoubleSide });
  }, 0x000000, 0.0, false, true);

  // Build exact instance color mapping for layout.json
  const instance_mapping = {
    "0": {
      entity_id: null,
      rgb: [0, 0, 0],
      label: "background"
    }
  };
  for (const [entId, instId] of Object.entries(idToInstance || {})) {
    if (instId === 0) continue;
    const hex = INSTANCE_PALETTE[(instId - 1) % INSTANCE_PALETTE.length];
    const r = (hex >> 16) & 255;
    const g = (hex >> 8) & 255;
    const b = hex & 255;
    instance_mapping[String(instId)] = {
      entity_id: entId,
      rgb: [r, g, b]
    };
  }

  // 6. Semantic Vis (Colorized semantic classes)
  const semantic_vis_png = renderToDataUrl(obj => {
    const classId = SEMANTIC_CLASSES[obj.userData.semantic] ?? 0;
    const col = SEMANTIC_COLORS[classId] ?? 0x111b25;
    return new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide });
  }, 0x111b25, 1.0, false, false);

  // 7. Semantic Raw (R encodes exact class ID, Alpha=0 for background)
  const semantic_raw_png = renderToDataUrl(obj => {
    const classId = SEMANTIC_CLASSES[obj.userData.semantic] ?? 0;
    const r = classId / 255.0;
    return new THREE.MeshBasicMaterial({ color: new THREE.Color(r, 0, 0), side: THREE.DoubleSide });
  }, 0x000000, 0.0, false, true);

  // 8. Planar Conditioning Map (Projected via Homography H, no 3D extrusion)
  let planar_png = null;
  if (planarContext && planarContext.geometry) {
    const pCanvas = renderPlanarCanvas(
      planarContext.geometry,
      planarContext.tags,
      planarContext.entityColors,
      planarContext.cameraData,
      planarContext.floorElevation ?? 0.0,
      width,
      height
    );
    planar_png = pCanvas.toDataURL('image/png');
  }

  // Restore proxy materials and line visibility on all meshes
  scene.traverse(obj => {
    if (obj.isLineSegments || obj.isLine) {
      if (obj.layers.isEnabled(1)) obj.visible = true;
    }
    if (obj.isMesh && obj.layers.isEnabled(1) && obj.userData.proxyMaterial) {
      obj.material = obj.userData.proxyMaterial;
    }
  });

  renderer.dispose();

  return {
    proxy_png,
    depth_vis_png,
    depth_raw_png,
    instance_vis_png,
    instance_raw_png,
    semantic_vis_png,
    semantic_raw_png,
    planar_png,
    instance_mapping
  };
}
