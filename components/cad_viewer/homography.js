/**
 * Planar Homography derivation and 2D canvas rendering for CAD floor plane.
 */

export function deriveHomography(camera, planeZ = 0.0, width = 1024, height = 1024) {
  const heading = ((camera.heading_deg || 0) * Math.PI) / 180;
  const pitch = ((camera.pitch_deg || 0) * Math.PI) / 180;
  const fovRad = ((camera.fov_deg || 60) * Math.PI) / 180;

  const forward = [
    Math.cos(pitch) * Math.cos(heading),
    Math.cos(pitch) * Math.sin(heading),
    Math.sin(pitch)
  ];
  const right = [
    Math.sin(heading),
    -Math.cos(heading),
    0.0
  ];
  const down = [
    forward[1] * right[2] - forward[2] * right[1],
    forward[2] * right[0] - forward[0] * right[2],
    forward[0] * right[1] - forward[1] * right[0]
  ];

  const pos = camera.position || [0, 0, 0];
  const t = [
    -(right[0] * pos[0] + right[1] * pos[1] + right[2] * pos[2]),
    -(down[0] * pos[0] + down[1] * pos[1] + down[2] * pos[2]),
    -(forward[0] * pos[0] + forward[1] * pos[1] + forward[2] * pos[2])
  ];

  const focal = width / (2 * Math.tan(fovRad / 2));
  const cx = width / 2;
  const cy = height / 2;

  // P = K * [R | t]
  const p1 = [focal * right[0] + cx * forward[0], focal * down[0] + cy * forward[0], forward[0]];
  const p2 = [focal * right[1] + cx * forward[1], focal * down[1] + cy * forward[1], forward[1]];
  const p3 = [focal * right[2] + cx * forward[2], focal * down[2] + cy * forward[2], forward[2]];
  const p4 = [focal * t[0] + cx * t[2], focal * t[1] + cy * t[2], t[2]];

  // H = [p1, p2, p3 * Z0 + p4]
  return [
    [p1[0], p2[0], p3[0] * planeZ + p4[0]],
    [p1[1], p2[1], p3[1] * planeZ + p4[1]],
    [p1[2], p2[2], p3[2] * planeZ + p4[2]]
  ];
}

export function clipPolygonToCameraFront(pts, H, nearW = 0.05) {
  if (!pts || pts.length < 3) return [];
  const [a, b, c] = H[2];
  const clipped = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const p1 = pts[i];
    const p2 = pts[(i + 1) % n];
    const val1 = a * p1[0] + b * p1[1] + c - nearW;
    const val2 = a * p2[0] + b * p2[1] + c - nearW;
    if (val1 >= 0) {
      clipped.push(p1);
      if (val2 < 0) {
        const factor = val1 / (val1 - val2);
        clipped.push([p1[0] + factor * (p2[0] - p1[0]), p1[1] + factor * (p2[1] - p1[1])]);
      }
    } else if (val2 >= 0) {
      const factor = val1 / (val1 - val2);
      clipped.push([p1[0] + factor * (p2[0] - p1[0]), p1[1] + factor * (p2[1] - p1[1])]);
    }
  }
  return clipped;
}

export function projectPolygon(pts, H, nearW = 0.05) {
  const clipped = clipPolygonToCameraFront(pts, H, nearW);
  if (clipped.length < 3) return [];
  const uvs = [];
  for (const p of clipped) {
    const uPrime = H[0][0] * p[0] + H[0][1] * p[1] + H[0][2];
    const vPrime = H[1][0] * p[0] + H[1][1] * p[1] + H[1][2];
    const wPrime = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2];
    if (wPrime > 0) {
      uvs.push([uPrime / wPrime, vPrime / wPrime]);
    }
  }
  return uvs;
}

export function clipLineSegmentToCameraFront(p1, p2, H, nearW = 0.05) {
  const [a, b, c] = H[2];
  const w1 = a * p1[0] + b * p1[1] + c;
  const w2 = a * p2[0] + b * p2[1] + c;
  if (w1 < nearW && w2 < nearW) return null;

  let pt1 = [p1[0], p1[1]];
  let pt2 = [p2[0], p2[1]];

  if (w1 >= nearW && w2 >= nearW) {
    // Both in front
  } else if (w1 >= nearW && w2 < nearW) {
    const t = (nearW - w1) / (w2 - w1);
    pt2 = [pt1[0] + t * (pt2[0] - pt1[0]), pt1[1] + t * (pt2[1] - pt1[1])];
  } else {
    const t = (nearW - w1) / (w2 - w1);
    pt1 = [pt1[0] + t * (pt2[0] - pt1[0]), pt1[1] + t * (pt2[1] - pt1[1])];
  }

  const proj = (p) => {
    const u = H[0][0] * p[0] + H[0][1] * p[1] + H[0][2];
    const v = H[1][0] * p[0] + H[1][1] * p[1] + H[1][2];
    const w = H[2][0] * p[0] + H[2][1] * p[1] + H[2][2];
    return [u / w, v / w];
  };

  return [proj(pt1), proj(pt2)];
}

function hexToRgba(color, alpha = 1.0) {
  let c = color;
  if (typeof c === 'number') c = '#' + c.toString(16).padStart(6, '0');
  if (typeof c === 'string' && c.startsWith('#')) {
    const num = parseInt(c.slice(1), 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return c;
}

export function renderPlanarCanvas(geometry, tags, entityColors, camera, planeZ, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  // Dark blueprint slate background #0b131e
  ctx.fillStyle = '#0b131e';
  ctx.fillRect(0, 0, width, height);

  const H = deriveHomography(camera, planeZ, width, height);

  const drawFillPoly = (uvs, fillStyle) => {
    if (!uvs || uvs.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(uvs[0][0], uvs[0][1]);
    for (let i = 1; i < uvs.length; i++) {
      ctx.lineTo(uvs[i][0], uvs[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
  };

  const drawSegment = (p1, p2, strokeStyle, strokeWidth = 1.5) => {
    const seg = clipLineSegmentToCameraFront(p1, p2, H);
    if (!seg) return;
    ctx.beginPath();
    ctx.moveTo(seg[0][0], seg[0][1]);
    ctx.lineTo(seg[1][0], seg[1][1]);
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.stroke();
  };

  const drawEntityLines = (ent, strokeStyle, strokeWidth = 1.5) => {
    for (const path of ent.paths || []) {
      for (let i = 1; i < path.length; i++) {
        drawSegment(path[i - 1], path[i], strokeStyle, strokeWidth);
      }
    }
  };

  // 1. Draw floor regions (subtle translucent floor fill + crisp teal boundary)
  for (const ent of geometry || []) {
    if (tags?.[ent.id] === 'floor') {
      for (const path of ent.paths || []) {
        const uvs = projectPolygon(path, H);
        drawFillPoly(uvs, 'rgba(26, 38, 57, 0.45)');
      }
      drawEntityLines(ent, '#65cbb7', 2);
    }
  }

  // 2. Draw wall footprints on floor plane (subtle fill + crisp wall color)
  for (const ent of geometry || []) {
    if (tags?.[ent.id] === 'wall') {
      for (const path of ent.paths || []) {
        const uvs = projectPolygon(path, H);
        drawFillPoly(uvs, 'rgba(251, 173, 101, 0.15)');
      }
      drawEntityLines(ent, '#fbad65', 2);
    }
  }

  // 3. Draw void / door thresholds on floor plane (subtle fill + cyan lines)
  for (const ent of geometry || []) {
    if (tags?.[ent.id] === 'void') {
      for (const path of ent.paths || []) {
        const uvs = projectPolygon(path, H);
        drawFillPoly(uvs, 'rgba(107, 191, 255, 0.15)');
      }
      drawEntityLines(ent, '#6bbfff', 2);
    }
  }

  // 4. Draw furniture: translucent fill for closed footprints + full CAD wireframe details
  for (const ent of geometry || []) {
    if (tags?.[ent.id] === 'furniture') {
      const col = entityColors?.[ent.id] || '#b99af7';
      const fillRgba = hexToRgba(col, 0.18);
      const strokeCol = typeof col === 'number' ? '#' + col.toString(16).padStart(6, '0') : col;

      for (const path of ent.paths || []) {
        const uvs = projectPolygon(path, H);
        drawFillPoly(uvs, fillRgba);
      }
      // Stroke ALL CAD lines (pillows, folds, chair arcs, etc.) with full opacity
      drawEntityLines(ent, strokeCol, 1.8);
    }
  }

  return canvas;
}

