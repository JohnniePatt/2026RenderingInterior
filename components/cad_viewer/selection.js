// Screen-space segment picking is stable at every zoom level and independent of GPU line width.
export function distanceToSegment(point, a, b) {
  const dx = b.x-a.x, dy = b.y-a.y;
  const length = dx*dx+dy*dy;
  const t = length ? Math.max(0, Math.min(1, ((point.x-a.x)*dx+(point.y-a.y)*dy)/length)) : 0;
  return Math.hypot(point.x-a.x-t*dx, point.y-a.y-t*dy);
}

export function pickEntity(point, entities, project) {
  let selected = null, nearest = 8;
  for (const entity of entities) {
    for (const path of entity.paths) {
      for (let i=1; i<path.length; i++) {
        const distance = distanceToSegment(point, project(path[i-1]), project(path[i]));
        if (distance < nearest) { nearest = distance; selected = entity.id; }
      }
    }
  }
  return selected;
}
