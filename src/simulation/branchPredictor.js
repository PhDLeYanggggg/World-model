import { createRng, range } from "./random.js";

const DEFAULT_WORLD = {
  width: 1120,
  height: 640,
  target: { x: 902, y: 510, radius: 38 },
};

const DEFAULT_MODEL = {
  trained: false,
  frameCount: 0,
  pairCount: 0,
  avgDt: 0.1,
  gravityY: 9.8,
  dampingX: 0.995,
  dampingY: 0.995,
  residual: {
    position: 5.5,
    velocity: 0.65,
  },
  loss: 0,
  confidence: 0.18,
  contactRate: 0,
  rewardRate: 0,
  status: "collecting data",
};

const DEFAULT_CONFIG = {
  horizon: 100,
  samplesPerStep: 12,
  localTopK: 3,
  beamWidth: 72,
  maxObjects: 34,
};

export function trainWorldModel(frames) {
  if (!Array.isArray(frames) || frames.length < 4) {
    return {
      ...DEFAULT_MODEL,
      frameCount: frames?.length ?? 0,
      status: "waiting for at least 4 recorded frames",
    };
  }

  const transitions = [];
  const dts = [];
  let contactCount = 0;
  let rewardCount = 0;

  for (let index = 1; index < frames.length; index += 1) {
    const previous = frames[index - 1];
    const current = frames[index];
    const dt = Math.max(0.016, Math.min(0.25, (current.timeMs - previous.timeMs) / 1000 || DEFAULT_MODEL.avgDt));
    const previousObjects = new Map(previous.state.objects.map((object) => [object.id, object]));

    dts.push(dt);
    contactCount += current.observation?.contacts?.length ?? 0;
    rewardCount += current.reward > previous.reward ? 1 : 0;

    current.state.objects.forEach((object) => {
      const before = previousObjects.get(object.id);
      if (!before) return;

      transitions.push({
        dt,
        before,
        after: object,
      });
    });
  }

  if (transitions.length < 8) {
    return {
      ...DEFAULT_MODEL,
      frameCount: frames.length,
      pairCount: transitions.length,
      avgDt: median(dts) || DEFAULT_MODEL.avgDt,
      status: "collecting more paired object transitions",
    };
  }

  const gravitySamples = [];
  let dampingXNumerator = 0;
  let dampingXDenominator = 0;
  let dampingYNumerator = 0;
  let dampingYDenominator = 0;

  transitions.forEach(({ before, after, dt }) => {
    const speedBefore = Math.hypot(before.velocity.x, before.velocity.y);
    const speedAfter = Math.hypot(after.velocity.x, after.velocity.y);

    if (speedBefore < 9 && speedAfter < 12) {
      gravitySamples.push((after.velocity.y - before.velocity.y) / dt);
    }

    dampingXNumerator += before.velocity.x * after.velocity.x;
    dampingXDenominator += before.velocity.x * before.velocity.x;
    dampingYNumerator += before.velocity.y * after.velocity.y;
    dampingYDenominator += before.velocity.y * before.velocity.y;
  });

  const avgDt = median(dts) || DEFAULT_MODEL.avgDt;
  const gravityY = clamp(median(gravitySamples), -24, 24) || DEFAULT_MODEL.gravityY;
  const dampingX = clamp(dampingXNumerator / Math.max(0.0001, dampingXDenominator), 0.82, 1.03);
  const dampingYRaw = dampingYNumerator / Math.max(0.0001, dampingYDenominator);
  const dampingY = clamp(Number.isFinite(dampingYRaw) ? dampingYRaw : DEFAULT_MODEL.dampingY, 0.82, 1.03);
  let squaredPositionError = 0;
  let squaredVelocityError = 0;

  transitions.forEach(({ before, after, dt }) => {
    const predictedVx = before.velocity.x * dampingX;
    const predictedVy = before.velocity.y * dampingY + gravityY * dt;
    const predictedX = before.position.x + predictedVx * dt;
    const predictedY = before.position.y + predictedVy * dt;

    squaredPositionError += (after.position.x - predictedX) ** 2 + (after.position.y - predictedY) ** 2;
    squaredVelocityError += (after.velocity.x - predictedVx) ** 2 + (after.velocity.y - predictedVy) ** 2;
  });

  const positionRmse = Math.sqrt(squaredPositionError / transitions.length);
  const velocityRmse = Math.sqrt(squaredVelocityError / transitions.length);
  const loss = Number((positionRmse * 0.12 + velocityRmse).toFixed(3));
  const confidenceFromPairs = 1 - Math.exp(-transitions.length / 450);
  const confidenceFromLoss = 1 / (1 + loss / 7);

  return {
    trained: true,
    frameCount: frames.length,
    pairCount: transitions.length,
    avgDt: Number(avgDt.toFixed(4)),
    gravityY: Number(gravityY.toFixed(4)),
    dampingX: Number(dampingX.toFixed(4)),
    dampingY: Number(dampingY.toFixed(4)),
    residual: {
      position: Number(clamp(positionRmse, 1.2, 28).toFixed(3)),
      velocity: Number(clamp(velocityRmse, 0.08, 4.8).toFixed(3)),
    },
    loss,
    confidence: Number(clamp(confidenceFromPairs * confidenceFromLoss, 0.05, 0.96).toFixed(3)),
    contactRate: Number((contactCount / frames.length).toFixed(3)),
    rewardRate: Number((rewardCount / frames.length).toFixed(3)),
    status: "online transition model trained",
  };
}

export function runBranchPrediction({ state, model = DEFAULT_MODEL, worldSpec = DEFAULT_WORLD, seed, config = {} }) {
  const options = { ...DEFAULT_CONFIG, ...config };
  const rng = createRng(seed || `${state.seed}:branch`);
  const objects = selectPredictionObjects(state.objects, options.maxObjects);

  let beam = [
    {
      state: {
        objects,
        reward: 0,
        impactCount: 0,
      },
      logProbability: 0,
      trace: [centerOfMass(objects)],
    },
  ];

  for (let step = 0; step < options.horizon; step += 1) {
    const expanded = [];

    beam.forEach((path) => {
      const candidates = generateBranchCandidates(model, rng, options.samplesPerStep)
        .sort((a, b) => b.probability - a.probability)
        .slice(0, options.localTopK);

      candidates.forEach((candidate) => {
        const nextState = stepSurrogateWorld(path.state, candidate, model, worldSpec);
        expanded.push({
          state: nextState,
          logProbability: path.logProbability + Math.log(candidate.probability),
          trace: appendTrace(path.trace, nextState.objects, step, options.horizon),
        });
      });
    });

    beam = expanded.sort((a, b) => b.logProbability - a.logProbability).slice(0, options.beamWidth);
  }

  const outcomes = clusterOutcomes(beam, worldSpec);
  const topPaths = normalizePaths(beam).slice(0, 12).map((path) => ({
    probability: path.probability,
    trace: path.trace,
    terminal: summarizeTerminalState(path.state, worldSpec),
  }));

  return {
    horizon: options.horizon,
    localTopK: options.localTopK,
    beamWidth: options.beamWidth,
    pathsAnalyzed: beam.length,
    modelConfidence: model.confidence,
    modelStatus: model.status,
    generatedAt: Date.now(),
    outcomes,
    topPaths,
  };
}

function selectPredictionObjects(objects, maxObjects) {
  return objects
    .map((object) => ({
      id: object.id,
      label: object.label,
      position: { ...object.position },
      velocity: { ...object.velocity },
      mass: object.mass,
      friction: object.friction,
      restitution: object.restitution,
      radius: object.label?.includes("particle") || object.label?.includes("node") ? 9 : 24,
    }))
    .sort((a, b) => objectPriority(b) - objectPriority(a))
    .slice(0, maxObjects);
}

function objectPriority(object) {
  const speed = Math.hypot(object.velocity.x, object.velocity.y);
  const materialBonus = object.label?.includes("particle") ? -5 : 0;
  return speed * 2 + object.mass * 0.2 + materialBonus;
}

function generateBranchCandidates(model, rng, samplesPerStep) {
  const candidates = [
    { z: 0, direction: 0, probability: 0.34 },
    { z: -0.7, direction: -1, probability: 0.23 },
    { z: 0.7, direction: 1, probability: 0.23 },
  ];

  while (candidates.length < samplesPerStep) {
    const z = range(rng, -2.4, 2.4);
    candidates.push({
      z,
      direction: rng() > 0.5 ? 1 : -1,
      probability: Math.exp(-0.5 * z * z) * (0.12 + rng() * 0.05),
    });
  }

  const normalizer = candidates.reduce((sum, candidate) => sum + candidate.probability, 0);
  return candidates.map((candidate) => ({
    ...candidate,
    gravity: model.gravityY + candidate.z * model.residual.velocity * 0.32,
    velocityNoise: candidate.z * model.residual.velocity * 0.18,
    positionNoise: candidate.z * model.residual.position * 0.09,
    probability: candidate.probability / normalizer,
  }));
}

function stepSurrogateWorld(pathState, branch, model, worldSpec) {
  const dt = model.avgDt || DEFAULT_MODEL.avgDt;
  const width = worldSpec.width ?? DEFAULT_WORLD.width;
  const height = worldSpec.height ?? DEFAULT_WORLD.height;
  let reward = pathState.reward;
  let impactCount = pathState.impactCount;

  const objects = pathState.objects.map((object, index) => {
    const phase = (index % 5) - 2;
    let vx = object.velocity.x * model.dampingX + branch.velocityNoise * phase * 0.18;
    let vy = object.velocity.y * model.dampingY + branch.gravity * dt + branch.velocityNoise * branch.direction;
    let x = object.position.x + vx * dt + branch.positionNoise * phase;
    let y = object.position.y + vy * dt + Math.abs(branch.positionNoise) * 0.4;
    const restitution = clamp(object.restitution || 0.25, 0.02, 0.92);
    const friction = clamp(object.friction || 0.35, 0, 1.4);

    if (x < object.radius) {
      x = object.radius;
      vx = Math.abs(vx) * restitution;
      impactCount += 1;
    } else if (x > width - object.radius) {
      x = width - object.radius;
      vx = -Math.abs(vx) * restitution;
      impactCount += 1;
    }

    if (y > height - object.radius) {
      y = height - object.radius;
      vy = -Math.abs(vy) * restitution;
      vx *= 1 - friction * 0.035;
      impactCount += 1;
    } else if (y < object.radius) {
      y = object.radius;
      vy = Math.abs(vy) * restitution;
      impactCount += 1;
    }

    if (insideTarget({ x, y }, worldSpec.target ?? DEFAULT_WORLD.target, object.radius)) {
      reward += 0.015;
      vx *= 0.992;
      vy *= 0.992;
    }

    return {
      ...object,
      position: {
        x: Number(x.toFixed(3)),
        y: Number(y.toFixed(3)),
      },
      velocity: {
        x: Number(vx.toFixed(3)),
        y: Number(vy.toFixed(3)),
      },
    };
  });

  return {
    objects,
    reward,
    impactCount,
  };
}

function clusterOutcomes(paths, worldSpec) {
  const normalized = normalizePaths(paths);
  const clusters = new Map();

  normalized.forEach((path) => {
    const terminal = summarizeTerminalState(path.state, worldSpec);
    const key = [
      Math.min(5, terminal.objectsInTarget),
      terminal.motionBucket,
      terminal.spreadBucket,
      terminal.centerBucket,
    ].join("|");

    const current = clusters.get(key) ?? {
      key,
      probability: 0,
      paths: 0,
      representative: terminal,
      bestProbability: 0,
    };

    current.probability += path.probability;
    current.paths += 1;
    if (path.probability > current.bestProbability) {
      current.bestProbability = path.probability;
      current.representative = terminal;
    }
    clusters.set(key, current);
  });

  return [...clusters.values()]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5)
    .map((cluster) => ({
      label: describeOutcome(cluster.representative),
      probability: Number(cluster.probability.toFixed(4)),
      paths: cluster.paths,
      representative: cluster.representative,
    }));
}

function summarizeTerminalState(pathState, worldSpec) {
  const objects = pathState.objects;
  const target = worldSpec.target ?? DEFAULT_WORLD.target;
  const center = centerOfMass(objects);
  const speeds = objects.map((object) => Math.hypot(object.velocity.x, object.velocity.y));
  const avgSpeed = average(speeds);
  const spread = average(objects.map((object) => Math.hypot(object.position.x - center.x, object.position.y - center.y)));
  const objectsInTarget = objects.filter((object) => insideTarget(object.position, target, object.radius)).length;

  return {
    objectsInTarget,
    avgSpeed: Number(avgSpeed.toFixed(3)),
    center: {
      x: Number(center.x.toFixed(1)),
      y: Number(center.y.toFixed(1)),
    },
    reward: Number(pathState.reward.toFixed(3)),
    impacts: pathState.impactCount,
    motionBucket: avgSpeed > 8 ? "active" : avgSpeed > 2.2 ? "drifting" : "settled",
    spreadBucket: spread > 260 ? "scattered" : spread > 140 ? "wide" : "compact",
    centerBucket: `${Math.floor(center.x / 180)}:${Math.floor(center.y / 130)}`,
  };
}

function describeOutcome(terminal) {
  if (terminal.objectsInTarget >= 5) {
    return "multiple objects accumulate in reward zone";
  }
  if (terminal.objectsInTarget > 0) {
    return "some objects reach reward zone";
  }
  if (terminal.motionBucket === "settled" && terminal.spreadBucket === "compact") {
    return "system settles into compact arrangement";
  }
  if (terminal.motionBucket === "active") {
    return "long-horizon motion remains active";
  }
  if (terminal.spreadBucket === "scattered") {
    return "objects scatter across lab";
  }
  return "objects drift without reward capture";
}

function normalizePaths(paths) {
  if (!paths.length) return [];
  const maxLog = Math.max(...paths.map((path) => path.logProbability));
  const weights = paths.map((path) => Math.exp(path.logProbability - maxLog));
  const total = weights.reduce((sum, weight) => sum + weight, 0) || 1;

  return paths
    .map((path, index) => ({
      ...path,
      probability: weights[index] / total,
    }))
    .sort((a, b) => b.probability - a.probability);
}

function appendTrace(trace, objects, step, horizon) {
  if (step % 10 !== 0 && step !== horizon - 1) return trace;
  return [...trace, centerOfMass(objects)];
}

function centerOfMass(objects) {
  if (!objects.length) return { x: 0, y: 0 };
  const totalMass = objects.reduce((sum, object) => sum + (object.mass || 1), 0) || objects.length;
  const x = objects.reduce((sum, object) => sum + object.position.x * (object.mass || 1), 0) / totalMass;
  const y = objects.reduce((sum, object) => sum + object.position.y * (object.mass || 1), 0) / totalMass;
  return { x, y };
}

function insideTarget(position, target, radius) {
  return Math.hypot(position.x - target.x, position.y - target.y) < target.radius + radius * 0.65;
}

function median(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return 0;
  return finite[Math.floor(finite.length / 2)];
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
