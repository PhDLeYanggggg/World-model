import Matter from "matter-js";
import { createRng, pick, range } from "./random.js";

const { Bodies, Body, Composite, Constraint, Events, Vector } = Matter;

const WALL = { fill: "#d8dde3", stroke: "#9ca6b2" };
const MATERIALS = {
  steel: { fill: "#778493", stroke: "#46515e" },
  rubber: { fill: "#2fa78f", stroke: "#13695a" },
  wood: { fill: "#c08a54", stroke: "#7a4e27" },
  glass: { fill: "#93c9e8", stroke: "#407697" },
  clay: { fill: "#c96f54", stroke: "#7d392a" },
  liquid: { fill: "#3aa6d8", stroke: "#176284" },
  sand: { fill: "#d5b66a", stroke: "#8a7134" },
};

let idCounter = 0;

export function buildWorld({ engine, seed, layers, randomization }) {
  Composite.clear(engine.world, false);
  idCounter = 0;
  const rng = createRng(`${seed}:${randomization.generation}`);
  const width = 1120;
  const height = 640;
  const walls = [
    wall(width / 2, height + 30, width, 60, "floor"),
    wall(width / 2, -30, width, 60, "ceiling"),
    wall(-30, height / 2, 60, height, "left-wall"),
    wall(width + 30, height / 2, 60, height, "right-wall"),
  ];

  Composite.add(engine.world, walls);
  engine.gravity.y = layers.rigid ? randomization.gravity : 0;

  const worldSpec = {
    width,
    height,
    camera: {
      x: 74,
      y: 84,
      angle: 0.26,
      fov: Math.PI * 0.48,
      range: 620,
      rayCount: 19,
    },
    light: {
      angle: randomization.lightAngle,
      intensity: randomization.lightIntensity,
    },
    target: { x: 902, y: 510, radius: 38 },
  };

  if (layers.environment) {
    addInteractiveEnvironment(engine, rng, randomization);
  }

  if (layers.rigid) {
    addRigidBodies(engine, rng, randomization);
  }

  if (layers.materials) {
    addMaterialApproximations(engine, rng, randomization);
  }

  attachCollisionEvents(engine, worldSpec);
  return worldSpec;
}

export function applyObjectPreset(body, preset) {
  const material = MATERIALS[preset.material] || MATERIALS.steel;
  Body.setMass(body, preset.mass);
  body.friction = preset.friction;
  body.frictionStatic = preset.frictionStatic ?? preset.friction * 1.8;
  body.restitution = preset.restitution;
  body.plugin.material = preset.material;
  body.render = { ...body.render, fillStyle: material.fill, strokeStyle: material.stroke };
}

export function spawnBody(engine, type, x, y, rng = Math.random) {
  const materialName = pick(rng, ["steel", "rubber", "wood", "glass", "clay"]);
  const size = range(rng, 30, 58);
  const body =
    type === "circle"
      ? Bodies.circle(x, y, size / 2, bodyOptions(`${materialName}-ball`, materialName, rng))
      : Bodies.rectangle(x, y, size * 1.15, size * 0.86, bodyOptions(`${materialName}-block`, materialName, rng));

  tag(body);
  Composite.add(engine.world, body);
  return body;
}

export function fractureBody(engine, body) {
  if (!body || body.plugin.fractured || !body.plugin.breakable) return [];
  body.plugin.fractured = true;
  const rng = Math.random;
  const { x, y } = body.position;
  const pieces = [];

  Composite.remove(engine.world, body);

  for (let index = 0; index < 7; index += 1) {
    const angle = (index / 7) * Math.PI * 2;
    const piece = Bodies.polygon(
      x + Math.cos(angle) * 12,
      y + Math.sin(angle) * 8,
      3 + (index % 3),
      18 + (index % 2) * 8,
      {
        ...bodyOptions("fracture-fragment", "glass", rng),
        friction: 0.42,
        restitution: 0.22,
      },
    );
    tag(piece, "fragment");
    Body.setVelocity(piece, {
      x: body.velocity.x + Math.cos(angle) * 3.2,
      y: body.velocity.y + Math.sin(angle) * 2.2,
    });
    Body.setAngularVelocity(piece, (index - 3) * 0.075);
    pieces.push(piece);
  }

  Composite.add(engine.world, pieces);
  return pieces;
}

function addRigidBodies(engine, rng, randomization) {
  const stack = [];
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4 - row; column += 1) {
      const block = Bodies.rectangle(
        186 + column * 48 + row * 24,
        546 - row * 42,
        range(rng, 40, 50),
        range(rng, 34, 42),
        bodyOptions("stack-block", pick(rng, ["wood", "steel", "rubber"]), rng, randomization),
      );
      tag(block);
      stack.push(block);
    }
  }

  const balls = Array.from({ length: 5 }, (_, index) => {
    const ball = Bodies.circle(
      440 + index * 48,
      104 + index * 8,
      range(rng, 18, 29),
      bodyOptions("dynamic-ball", pick(rng, ["rubber", "steel", "glass"]), rng, randomization),
    );
    tag(ball);
    return ball;
  });

  Composite.add(engine.world, [...stack, ...balls]);
}

function addInteractiveEnvironment(engine, rng, randomization) {
  const ramp = Bodies.rectangle(420, 512, 250, 22, {
    isStatic: true,
    angle: -0.32,
    friction: 0.9,
    render: { fillStyle: "#d4dde5", strokeStyle: "#8f9ba8" },
    label: "inclined-ramp",
  });

  const containerParts = [
    Bodies.rectangle(872, 576, 180, 18, { isStatic: true, label: "container-floor" }),
    Bodies.rectangle(786, 524, 18, 120, { isStatic: true, label: "container-left" }),
    Bodies.rectangle(958, 524, 18, 120, { isStatic: true, label: "container-right" }),
  ];
  containerParts.forEach((part) => {
    part.render.fillStyle = "#d7e0e7";
    part.render.strokeStyle = "#8d99a6";
    tag(part, "fixture");
  });

  const anchor = Bodies.circle(660, 130, 8, {
    isStatic: true,
    render: { fillStyle: "#1f2937", strokeStyle: "#111827" },
    label: "hinge-anchor",
  });
  const pendulum = Bodies.rectangle(704, 274, 26, 148, bodyOptions("hinged-link", "steel", rng, randomization));
  tag(anchor, "fixture");
  tag(pendulum);
  const hinge = Constraint.create({
    bodyA: anchor,
    pointA: { x: 0, y: 0 },
    bodyB: pendulum,
    pointB: { x: 0, y: -66 },
    length: 0,
    stiffness: 0.94,
    damping: 0.012,
    label: "hinge-constraint",
  });

  const ropeBodies = [];
  const ropeConstraints = [];
  let previous = Bodies.circle(972, 108, 7, {
    isStatic: true,
    render: { fillStyle: "#1f2937", strokeStyle: "#111827" },
    label: "rope-anchor",
  });
  tag(previous, "fixture");
  ropeBodies.push(previous);

  for (let index = 0; index < 8; index += 1) {
    const bead = Bodies.circle(972 + index * 12, 138 + index * 26, 9, {
      ...bodyOptions("rope-link", "wood", rng, randomization),
      frictionAir: 0.012,
    });
    tag(bead);
    ropeBodies.push(bead);
    ropeConstraints.push(
      Constraint.create({
        bodyA: previous,
        bodyB: bead,
        length: 27,
        stiffness: 0.78,
        damping: 0.035,
        label: "rope-constraint",
      }),
    );
    previous = bead;
  }

  const breakable = Bodies.rectangle(604, 448, 108, 28, {
    ...bodyOptions("breakable-slab", "glass", rng, randomization),
    restitution: 0.16,
  });
  tag(breakable);
  breakable.plugin.breakable = true;

  Composite.add(engine.world, [ramp, ...containerParts, anchor, pendulum, hinge, ...ropeBodies, ...ropeConstraints, breakable]);
}

function addMaterialApproximations(engine, rng, randomization) {
  const particles = [];

  for (let index = 0; index < 42; index += 1) {
    const sand = Bodies.circle(824 + (index % 7) * 18, 178 + Math.floor(index / 7) * 18, range(rng, 5, 8), {
      ...bodyOptions("granular-particle", "sand", rng, randomization),
      friction: 0.82,
      frictionAir: 0.018,
      restitution: 0.05,
    });
    tag(sand, "particle");
    particles.push(sand);
  }

  for (let index = 0; index < 24; index += 1) {
    const drop = Bodies.circle(722 + (index % 6) * 15, 224 + Math.floor(index / 6) * 16, range(rng, 5, 7), {
      ...bodyOptions("liquid-particle", "liquid", rng, randomization),
      friction: 0.02,
      frictionAir: 0.035,
      restitution: 0.04,
      density: 0.0009,
    });
    tag(drop, "particle");
    particles.push(drop);
  }

  const soft = [];
  const constraints = [];
  const columns = 4;
  const rows = 3;
  const startX = 540;
  const startY = 190;
  const spacing = 28;

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const node = Bodies.circle(startX + column * spacing, startY + row * spacing, 10, {
        ...bodyOptions("soft-body-node", "clay", rng, randomization),
        frictionAir: 0.018,
        restitution: 0.08,
      });
      tag(node, "soft-node");
      soft.push(node);
    }
  }

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const current = soft[row * columns + column];
      const right = column < columns - 1 ? soft[row * columns + column + 1] : null;
      const down = row < rows - 1 ? soft[(row + 1) * columns + column] : null;
      if (right) constraints.push(softConstraint(current, right));
      if (down) constraints.push(softConstraint(current, down));
    }
  }

  Composite.add(engine.world, [...particles, ...soft, ...constraints]);
}

function softConstraint(bodyA, bodyB) {
  return Constraint.create({
    bodyA,
    bodyB,
    length: 28,
    stiffness: 0.34,
    damping: 0.12,
    label: "soft-constraint",
    render: { strokeStyle: "#b65b45" },
  });
}

function attachCollisionEvents(engine, worldSpec) {
  engine.plugin.worldLabEvents = {
    contacts: [],
    causalEvents: [],
    reward: 0,
  };

  Events.on(engine, "collisionStart", (event) => {
    event.pairs.forEach((pair) => {
      const bodyA = pair.bodyA;
      const bodyB = pair.bodyB;
      const contact = {
        t: Math.round(engine.timing.timestamp),
        a: bodyA.plugin.worldId ?? bodyA.label,
        b: bodyB.plugin.worldId ?? bodyB.label,
        normal: {
          x: Number(pair.collision.normal.x.toFixed(2)),
          y: Number(pair.collision.normal.y.toFixed(2)),
        },
      };
      engine.plugin.worldLabEvents.contacts.push(contact);

      const speed = Vector.magnitude(Vector.sub(bodyA.velocity, bodyB.velocity));
      const breakable = bodyA.plugin.breakable ? bodyA : bodyB.plugin.breakable ? bodyB : null;
      if (breakable && speed > 5.6) {
        const fragments = fractureBody(engine, breakable);
        if (fragments.length) {
          engine.plugin.worldLabEvents.causalEvents.push({
            t: Math.round(engine.timing.timestamp),
            type: "fracture",
            cause: "collision_velocity_threshold",
            objectId: breakable.plugin.worldId,
            fragments: fragments.map((fragment) => fragment.plugin.worldId),
          });
        }
      }

      [bodyA, bodyB].forEach((body) => {
        if (!body.isStatic && distance(body.position, worldSpec.target) < worldSpec.target.radius + 28) {
          engine.plugin.worldLabEvents.reward += 1;
          engine.plugin.worldLabEvents.causalEvents.push({
            t: Math.round(engine.timing.timestamp),
            type: "reward",
            cause: "object_inside_target_zone",
            objectId: body.plugin.worldId,
            value: 1,
          });
        }
      });
    });

    engine.plugin.worldLabEvents.contacts = engine.plugin.worldLabEvents.contacts.slice(-80);
    engine.plugin.worldLabEvents.causalEvents = engine.plugin.worldLabEvents.causalEvents.slice(-80);
  });
}

function bodyOptions(label, materialName, rng, randomization = {}) {
  const material = MATERIALS[materialName] || MATERIALS.steel;
  const jitter = randomization.enabled ? range(rng, 0.82, 1.18) : 1;
  return {
    label,
    friction: range(rng, 0.18, 0.72) * jitter,
    frictionStatic: range(rng, 0.28, 1.1) * jitter,
    restitution: range(rng, 0.04, 0.68),
    density: range(rng, 0.00075, 0.0026) * jitter,
    render: { fillStyle: material.fill, strokeStyle: material.stroke },
    plugin: { material: materialName },
  };
}

function wall(x, y, width, height, label) {
  const body = Bodies.rectangle(x, y, width, height, {
    isStatic: true,
    label,
    render: { fillStyle: WALL.fill, strokeStyle: WALL.stroke },
  });
  tag(body, "boundary");
  return body;
}

function tag(body, kind = "body") {
  body.plugin = body.plugin || {};
  body.plugin.worldId = body.plugin.worldId || `${kind}-${String(++idCounter).padStart(3, "0")}`;
  body.plugin.kind = kind;
  return body;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
