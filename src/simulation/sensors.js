import Matter from "matter-js";

const { Query, Vector } = Matter;

export function sampleSensors({ engine, worldSpec, selectedBody, contacts, causalEvents, reward }) {
  const bodies = engine.world.bodies.filter((body) => !body.isStatic && !body.isSensor);
  const camera = worldSpec.camera;
  const rays = [];
  const visibleIds = new Set();

  for (let index = 0; index < camera.rayCount; index += 1) {
    const t = camera.rayCount === 1 ? 0.5 : index / (camera.rayCount - 1);
    const angle = camera.angle - camera.fov / 2 + camera.fov * t;
    const end = {
      x: camera.x + Math.cos(angle) * camera.range,
      y: camera.y + Math.sin(angle) * camera.range,
    };
    const hits = Query.ray(bodies, camera, end, 2);
    const hit = hits
      .map((collision) => collision.body || collision.bodyA)
      .filter(Boolean)
      .sort((a, b) => Vector.magnitude(Vector.sub(a.position, camera)) - Vector.magnitude(Vector.sub(b.position, camera)))[0];

    if (hit) {
      visibleIds.add(hit.plugin.worldId);
    }

    rays.push({
      index,
      end: hit ? { x: hit.position.x, y: hit.position.y } : end,
      distance: hit ? Number(Vector.magnitude(Vector.sub(hit.position, camera)).toFixed(2)) : camera.range,
      hitId: hit?.plugin.worldId ?? null,
      occluded: hits.length > 1,
    });
  }

  const selected = selectedBody
    ? {
        id: selectedBody.plugin.worldId,
        label: selectedBody.label,
        position: roundVec(selectedBody.position),
        velocity: roundVec(selectedBody.velocity),
        angle: Number(selectedBody.angle.toFixed(3)),
        angularVelocity: Number(selectedBody.angularVelocity.toFixed(3)),
        mass: Number(selectedBody.mass.toFixed(3)),
        friction: Number(selectedBody.friction.toFixed(3)),
        restitution: Number(selectedBody.restitution.toFixed(3)),
      }
    : null;

  return {
    camera: {
      origin: { x: camera.x, y: camera.y },
      fov: Number(camera.fov.toFixed(3)),
      range: camera.range,
      rays,
      visibleIds: [...visibleIds],
    },
    selected,
    velocities: bodies.slice(0, 20).map((body) => ({
      id: body.plugin.worldId,
      speed: Number(Vector.magnitude(body.velocity).toFixed(3)),
    })),
    contacts: contacts.slice(-12),
    occlusionCount: rays.filter((ray) => ray.occluded).length,
    reward: Number(reward.toFixed(3)),
    causalEvents: causalEvents.slice(-12),
  };
}

export function serializeState(engine, timeMs, seed) {
  return {
    timeMs: Math.round(timeMs),
    seed,
    gravity: {
      x: Number(engine.gravity.x.toFixed(3)),
      y: Number(engine.gravity.y.toFixed(3)),
      scale: engine.gravity.scale,
    },
    objects: engine.world.bodies
      .filter((body) => !body.isStatic && !body.isSensor)
      .map((body) => ({
        id: body.plugin.worldId,
        label: body.label,
        position: roundVec(body.position),
        velocity: roundVec(body.velocity),
        angle: Number(body.angle.toFixed(3)),
        angularVelocity: Number(body.angularVelocity.toFixed(3)),
        mass: Number(body.mass.toFixed(3)),
        friction: Number(body.friction.toFixed(3)),
        restitution: Number(body.restitution.toFixed(3)),
      })),
  };
}

function roundVec(vector) {
  return {
    x: Number(vector.x.toFixed(3)),
    y: Number(vector.y.toFixed(3)),
  };
}
