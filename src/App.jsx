import { useEffect, useMemo, useRef, useState } from "react";
import Matter from "matter-js";
import {
  Activity,
  Box,
  Camera,
  Circle,
  Database,
  Download,
  FlaskConical,
  Gauge,
  Hand,
  Pause,
  Play,
  Radar,
  RefreshCcw,
  RotateCcw,
  ScanLine,
  Shuffle,
  Square,
  StepForward,
  Target,
  Waves,
} from "lucide-react";
import { buildWorld, spawnBody } from "./simulation/createWorld.js";
import { createRng } from "./simulation/random.js";
import { sampleSensors, serializeState } from "./simulation/sensors.js";

const { Body, Composite, Constraint, Engine, Query, Vector } = Matter;

const INITIAL_LAYERS = {
  rigid: true,
  sensors: true,
  environment: true,
  materials: true,
  randomization: true,
  export: true,
};

const LAYER_CONFIG = [
  {
    key: "rigid",
    title: "Newtonian Rigid Bodies",
    detail: "gravity, collision, friction, elasticity, constraints",
    icon: Box,
  },
  {
    key: "sensors",
    title: "Sensors & Observations",
    detail: "camera rays, depth, velocity, contacts, occlusion",
    icon: Camera,
  },
  {
    key: "environment",
    title: "Interactive Environment",
    detail: "push, pull, stack, ramp, container, rope, hinge",
    icon: Hand,
  },
  {
    key: "materials",
    title: "Material Approximations",
    detail: "soft lattice, fracture, liquid-like drops, particles",
    icon: Waves,
  },
  {
    key: "randomization",
    title: "Domain Randomization",
    detail: "mass, friction, light, color, shape, seed",
    icon: Shuffle,
  },
  {
    key: "export",
    title: "Training Data Export",
    detail: "state, action, observation, reward, causal events",
    icon: Database,
  },
];

const MODES = [
  { id: "select", label: "Select", icon: ScanLine },
  { id: "drag", label: "Drag", icon: Hand },
  { id: "push", label: "Push", icon: Target },
  { id: "pull", label: "Pull", icon: Radar },
];

function makeRandomization(seed, generation, enabled = true) {
  const rng = createRng(`${seed}:domain:${generation}`);
  return {
    enabled,
    generation,
    gravity: Number((0.78 + rng() * 0.54).toFixed(2)),
    lightAngle: Number((-0.7 + rng() * 1.4).toFixed(2)),
    lightIntensity: Number((0.65 + rng() * 0.35).toFixed(2)),
  };
}

function App() {
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const worldSpecRef = useRef(null);
  const animationRef = useRef(null);
  const selectedRef = useRef(null);
  const dragConstraintRef = useRef(null);
  const pointerRef = useRef(null);
  const datasetRef = useRef([]);
  const lastActionRef = useRef({ type: "none" });
  const frameIndexRef = useRef(0);
  const episodeStartRef = useRef(performance.now());
  const runningRef = useRef(true);
  const recordingRef = useRef(true);
  const layersRef = useRef(INITIAL_LAYERS);
  const modeRef = useRef("drag");
  const seedRef = useRef("world-seed-001");
  const rngRef = useRef(createRng("world-seed-001"));

  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [mode, setMode] = useState("drag");
  const [running, setRunning] = useState(true);
  const [recording, setRecording] = useState(true);
  const [seedInput, setSeedInput] = useState("world-seed-001");
  const [worldSeed, setWorldSeed] = useState("world-seed-001");
  const [randomization, setRandomization] = useState(() => makeRandomization("world-seed-001", 0));
  const [telemetry, setTelemetry] = useState(null);
  const [selectedSnapshot, setSelectedSnapshot] = useState(null);
  const [episodeStats, setEpisodeStats] = useState({ frames: 0, timeMs: 0 });
  const [exportStatus, setExportStatus] = useState("ready");

  const activeLayerCount = useMemo(() => Object.values(layers).filter(Boolean).length, [layers]);

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    recordingRef.current = recording;
  }, [recording]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    layersRef.current = layers;
  }, [layers]);

  useEffect(() => {
    seedRef.current = worldSeed;
    rngRef.current = createRng(`${worldSeed}:${randomization.generation}:spawns`);
  }, [worldSeed, randomization.generation]);

  useEffect(() => {
    const engine = Engine.create({
      enableSleeping: false,
      constraintIterations: 4,
      positionIterations: 8,
      velocityIterations: 8,
    });

    engineRef.current = engine;
    resetWorld(engine, worldSeed, layers, randomization);

    const tick = () => {
      const currentEngine = engineRef.current;
      const canvas = canvasRef.current;
      const worldSpec = worldSpecRef.current;

      if (currentEngine && canvas && worldSpec) {
        if (runningRef.current) {
          Engine.update(currentEngine, 1000 / 60);
          frameIndexRef.current += 1;
        }

        const events = currentEngine.plugin.worldLabEvents || { contacts: [], causalEvents: [], reward: 0 };
        const observation = layersRef.current.sensors
          ? sampleSensors({
              engine: currentEngine,
              worldSpec,
              selectedBody: selectedRef.current,
              contacts: events.contacts,
              causalEvents: events.causalEvents,
              reward: events.reward,
            })
          : null;

        drawWorld(canvas, currentEngine, worldSpec, {
          selectedBody: selectedRef.current,
          observation,
          layers: layersRef.current,
          pointer: pointerRef.current,
          seed: seedRef.current,
          running: runningRef.current,
          recording: recordingRef.current,
        });

        if (runningRef.current && recordingRef.current && layersRef.current.export && frameIndexRef.current % 6 === 0) {
          const timeMs = performance.now() - episodeStartRef.current;
          const frame = {
            episodeId: `${seedRef.current}-${randomization.generation}`,
            step: datasetRef.current.length,
            timeMs: Math.round(timeMs),
            action: lastActionRef.current,
            state: serializeState(currentEngine, timeMs, seedRef.current),
            observation,
            reward: observation?.reward ?? 0,
            causalEvents: observation?.causalEvents ?? [],
          };

          datasetRef.current.push(frame);
          lastActionRef.current = { type: "none" };
        }

        if (frameIndexRef.current % 8 === 0) {
          setTelemetry(observation);
          setSelectedSnapshot(observation?.selected ?? null);
          setEpisodeStats({
            frames: datasetRef.current.length,
            timeMs: performance.now() - episodeStartRef.current,
          });
        }
      }

      animationRef.current = requestAnimationFrame(tick);
    };

    animationRef.current = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(animationRef.current);
      Composite.clear(engine.world, false);
      Engine.clear(engine);
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!engineRef.current) return;
    resetWorld(engineRef.current, worldSeed, layers, randomization);
  }, [worldSeed, layers, randomization]);

  function resetWorld(engine = engineRef.current, seed = worldSeed, nextLayers = layers, nextRandomization = randomization) {
    if (!engine) return;
    selectedRef.current = null;
    dragConstraintRef.current = null;
    pointerRef.current = null;
    datasetRef.current = [];
    lastActionRef.current = { type: "reset_world" };
    frameIndexRef.current = 0;
    episodeStartRef.current = performance.now();
    worldSpecRef.current = buildWorld({
      engine,
      seed,
      layers: nextLayers,
      randomization: nextRandomization,
    });
    setExportStatus("ready");
  }

  function handleCanvasPointerDown(event) {
    const engine = engineRef.current;
    if (!engine) return;
    const point = canvasPoint(event);
    pointerRef.current = point;
    const body = pickBodyAt(engine, point);
    selectedRef.current = body;

    if (!body) {
      setSelectedSnapshot(null);
      lastActionRef.current = { type: "select_empty", point: roundPoint(point) };
      return;
    }

    if (modeRef.current === "drag" && !body.isStatic) {
      const constraint = Constraint.create({
        pointA: point,
        bodyB: body,
        pointB: Vector.sub(point, body.position),
        stiffness: 0.12,
        damping: 0.08,
        label: "pointer-drag-constraint",
      });
      dragConstraintRef.current = constraint;
      Composite.add(engine.world, constraint);
      lastActionRef.current = { type: "drag_start", objectId: body.plugin.worldId, point: roundPoint(point) };
    } else if ((modeRef.current === "push" || modeRef.current === "pull") && !body.isStatic) {
      applyImpulse(body, point, modeRef.current);
      lastActionRef.current = { type: modeRef.current, objectId: body.plugin.worldId, point: roundPoint(point) };
    } else {
      lastActionRef.current = { type: "select", objectId: body.plugin.worldId, point: roundPoint(point) };
    }
  }

  function handleCanvasPointerMove(event) {
    const point = canvasPoint(event);
    pointerRef.current = point;
    if (dragConstraintRef.current) {
      dragConstraintRef.current.pointA = point;
      lastActionRef.current = {
        type: "drag",
        objectId: dragConstraintRef.current.bodyB?.plugin.worldId,
        point: roundPoint(point),
      };
    }
  }

  function handleCanvasPointerUp() {
    const engine = engineRef.current;
    if (dragConstraintRef.current && engine) {
      lastActionRef.current = {
        type: "drag_end",
        objectId: dragConstraintRef.current.bodyB?.plugin.worldId,
      };
      Composite.remove(engine.world, dragConstraintRef.current);
      dragConstraintRef.current = null;
    }
    pointerRef.current = null;
  }

  function updateSelectedProperty(property, value) {
    const body = selectedRef.current;
    if (!body || body.isStatic) return;
    const numericValue = Number(value);

    if (property === "mass") {
      Body.setMass(body, numericValue);
    } else if (property === "friction") {
      body.friction = numericValue;
      body.frictionStatic = numericValue * 1.8;
    } else if (property === "restitution") {
      body.restitution = numericValue;
    }

    lastActionRef.current = {
      type: "set_property",
      objectId: body.plugin.worldId,
      property,
      value: numericValue,
    };
    setSelectedSnapshot({
      ...selectedSnapshot,
      [property]: numericValue,
    });
  }

  function canvasPoint(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const worldSpec = worldSpecRef.current || { width: canvas.width, height: canvas.height };
    return {
      x: ((event.clientX - rect.left) / rect.width) * worldSpec.width,
      y: ((event.clientY - rect.top) / rect.height) * worldSpec.height,
    };
  }

  function pickBodyAt(engine, point) {
    const candidates = Query.point(
      engine.world.bodies.filter((body) => body.plugin.kind !== "boundary"),
      point,
    );
    return candidates.sort((a, b) => Number(a.isStatic) - Number(b.isStatic))[0] ?? null;
  }

  function spawn(type) {
    const engine = engineRef.current;
    if (!engine) return;
    const rng = rngRef.current;
    const body = spawnBody(engine, type, 180 + rng() * 260, 84 + rng() * 60, rng);
    selectedRef.current = body;
    lastActionRef.current = { type: `spawn_${type}`, objectId: body.plugin.worldId };
  }

  function stepOnce() {
    const engine = engineRef.current;
    if (!engine) return;
    Engine.update(engine, 1000 / 60);
    frameIndexRef.current += 1;
    lastActionRef.current = { type: "step_once" };
    setRunning(false);
  }

  function randomizeDomain() {
    const nextGeneration = randomization.generation + 1;
    const next = makeRandomization(worldSeed, nextGeneration, layers.randomization);
    setRandomization(next);
    lastActionRef.current = { type: "domain_randomize", generation: nextGeneration };
  }

  function applySeedReset() {
    setWorldSeed(seedInput.trim() || "world-seed-001");
    lastActionRef.current = { type: "seed_reset", seed: seedInput };
  }

  function exportJsonl() {
    if (!datasetRef.current.length) {
      setExportStatus("no frames recorded");
      return;
    }

    const lines = datasetRef.current.map((frame) => JSON.stringify(frame)).join("\n");
    const blob = new Blob([`${lines}\n`], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `world-lab-${worldSeed}-${Date.now()}.jsonl`;
    anchor.click();
    URL.revokeObjectURL(url);
    setExportStatus(`${datasetRef.current.length} frames exported`);
  }

  const selected = selectedSnapshot;
  const contacts = telemetry?.contacts ?? [];
  const causalEvents = telemetry?.causalEvents ?? [];
  const speedBars = telemetry?.velocities?.slice(0, 12) ?? [];

  return (
    <div className="appShell">
      <header className="topbar">
        <div className="brand">
          <div className="brandMark">
            <FlaskConical size={18} />
          </div>
          <div>
            <h1>World Lab</h1>
            <p>Bounded physics world for world-model training</p>
          </div>
        </div>

        <div className="topMetrics">
          <Metric icon={Activity} label="Layers" value={`${activeLayerCount}/6`} />
          <Metric icon={Gauge} label="Gravity" value={`${randomization.gravity.toFixed(2)}g`} />
          <Metric icon={Database} label="Frames" value={episodeStats.frames} />
        </div>
      </header>

      <main className="workbench">
        <aside className="leftPanel">
          <section className="panelSection seedSection">
            <label className="fieldLabel" htmlFor="seed">
              Seed
            </label>
            <div className="seedRow">
              <input id="seed" value={seedInput} onChange={(event) => setSeedInput(event.target.value)} />
              <IconButton title="Reset with seed" onClick={applySeedReset}>
                <RefreshCcw size={16} />
              </IconButton>
            </div>
          </section>

          <section className="panelSection">
            <div className="sectionHeader">
              <span>World Layers</span>
              <span className="miniStat">{activeLayerCount} active</span>
            </div>
            <div className="layerList">
              {LAYER_CONFIG.map((layer) => (
                <LayerToggle
                  key={layer.key}
                  layer={layer}
                  enabled={layers[layer.key]}
                  onChange={() => setLayers((current) => ({ ...current, [layer.key]: !current[layer.key] }))}
                />
              ))}
            </div>
          </section>

          <section className="panelSection">
            <div className="sectionHeader">
              <span>Interaction</span>
              <span className="miniStat">{mode}</span>
            </div>
            <div className="modeGrid">
              {MODES.map((item) => (
                <button
                  className={`modeButton ${mode === item.id ? "active" : ""}`}
                  key={item.id}
                  onClick={() => setMode(item.id)}
                  title={item.label}
                  type="button"
                >
                  <item.icon size={16} />
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
            <div className="spawnRow">
              <button onClick={() => spawn("circle")} type="button">
                <Circle size={15} />
                Ball
              </button>
              <button onClick={() => spawn("block")} type="button">
                <Square size={15} />
                Block
              </button>
            </div>
          </section>

          <section className="panelSection">
            <div className="sectionHeader">
              <span>Domain</span>
              <span className="miniStat">gen {randomization.generation}</span>
            </div>
            <label className="rangeLabel">
              Gravity
              <input
                min="0"
                max="1.8"
                step="0.01"
                type="range"
                value={randomization.gravity}
                onChange={(event) => {
                  const gravity = Number(event.target.value);
                  setRandomization((current) => ({ ...current, gravity }));
                  if (engineRef.current) engineRef.current.gravity.y = gravity;
                }}
              />
              <span>{randomization.gravity.toFixed(2)}g</span>
            </label>
            <button className="fullButton" disabled={!layers.randomization} onClick={randomizeDomain} type="button">
              <Shuffle size={15} />
              Randomize world
            </button>
          </section>
        </aside>

        <section className="simulationPane">
          <div className="canvasToolbar">
            <div className="toolbarGroup">
              <IconButton title={running ? "Pause" : "Play"} onClick={() => setRunning((value) => !value)}>
                {running ? <Pause size={16} /> : <Play size={16} />}
              </IconButton>
              <IconButton title="Step one frame" onClick={stepOnce}>
                <StepForward size={16} />
              </IconButton>
              <IconButton title="Reset world" onClick={() => resetWorld()}>
                <RotateCcw size={16} />
              </IconButton>
            </div>
            <div className="canvasStatus">
              <span>{running ? "running" : "paused"}</span>
              <span>{recording && layers.export ? "recording" : "not recording"}</span>
              <span>{Math.round(episodeStats.timeMs / 100) / 10}s</span>
            </div>
          </div>
          <canvas
            ref={canvasRef}
            width="1120"
            height="640"
            onPointerDown={handleCanvasPointerDown}
            onPointerMove={handleCanvasPointerMove}
            onPointerUp={handleCanvasPointerUp}
            onPointerLeave={handleCanvasPointerUp}
          />
        </section>

        <aside className="rightPanel">
          <section className="panelSection">
            <div className="sectionHeader">
              <span>Inspector</span>
              <span className="miniStat">{selected?.id ?? "none"}</span>
            </div>
            {selected ? (
              <div className="inspectorCard">
                <div className="objectTitle">
                  <span>{selected.label}</span>
                  <code>{selected.id}</code>
                </div>
                <Readout label="Position" value={`${selected.position.x}, ${selected.position.y}`} />
                <Readout label="Velocity" value={`${selected.velocity.x}, ${selected.velocity.y}`} />
                <RangeControl
                  label="Mass"
                  min={0.2}
                  max={28}
                  step={0.1}
                  value={selected.mass}
                  onChange={(value) => updateSelectedProperty("mass", value)}
                />
                <RangeControl
                  label="Friction"
                  min={0}
                  max={1.4}
                  step={0.01}
                  value={selected.friction}
                  onChange={(value) => updateSelectedProperty("friction", value)}
                />
                <RangeControl
                  label="Restitution"
                  min={0}
                  max={1}
                  step={0.01}
                  value={selected.restitution}
                  onChange={(value) => updateSelectedProperty("restitution", value)}
                />
              </div>
            ) : (
              <div className="emptyState">Select or drag an object in the world.</div>
            )}
          </section>

          <section className="panelSection">
            <div className="sectionHeader">
              <span>Sensor Readout</span>
              <span className="miniStat">{layers.sensors ? "live" : "off"}</span>
            </div>
            <div className="sensorGrid">
              <Readout label="Visible IDs" value={telemetry?.camera?.visibleIds?.length ?? 0} />
              <Readout label="Depth Rays" value={telemetry?.camera?.rays?.length ?? 0} />
              <Readout label="Occlusions" value={telemetry?.occlusionCount ?? 0} />
              <Readout label="Reward" value={(telemetry?.reward ?? 0).toFixed(1)} />
            </div>
            <div className="miniChart" aria-label="velocity chart">
              {speedBars.map((item) => (
                <span
                  key={item.id}
                  style={{ height: `${Math.min(100, 12 + item.speed * 16)}%` }}
                  title={`${item.id}: ${item.speed}`}
                />
              ))}
            </div>
          </section>

          <section className="panelSection eventPanel">
            <div className="sectionHeader">
              <span>Contacts</span>
              <span className="miniStat">{contacts.length}</span>
            </div>
            <EventList events={contacts} empty="No contacts yet." />
          </section>

          <section className="panelSection eventPanel">
            <div className="sectionHeader">
              <span>Causal Events</span>
              <span className="miniStat">{causalEvents.length}</span>
            </div>
            <EventList events={causalEvents} empty="Awaiting fracture or reward events." />
          </section>
        </aside>
      </main>

      <footer className="recorder">
        <div className="recorderLeft">
          <button
            className={`recordButton ${recording ? "active" : ""}`}
            disabled={!layers.export}
            onClick={() => setRecording((value) => !value)}
            type="button"
          >
            <Database size={15} />
            {recording ? "Recording JSONL" : "Record paused"}
          </button>
          <span>episode {worldSeed}</span>
          <span>{episodeStats.frames} frames</span>
          <span>{exportStatus}</span>
        </div>
        <div className="recorderRight">
          <code>
            {"{ state, action, observation, reward, causalEvents }"}
          </code>
          <button className="exportButton" disabled={!layers.export} onClick={exportJsonl} type="button">
            <Download size={15} />
            Export JSONL
          </button>
        </div>
      </footer>
    </div>
  );
}

function LayerToggle({ layer, enabled, onChange }) {
  const Icon = layer.icon;
  return (
    <button className={`layerToggle ${enabled ? "enabled" : ""}`} onClick={onChange} type="button">
      <Icon size={16} />
      <span>
        <strong>{layer.title}</strong>
        <small>{layer.detail}</small>
      </span>
      <i>{enabled ? "on" : "off"}</i>
    </button>
  );
}

function IconButton({ children, title, onClick }) {
  return (
    <button className="iconButton" onClick={onClick} title={title} type="button" aria-label={title}>
      {children}
    </button>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric">
      <Icon size={15} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Readout({ label, value }) {
  return (
    <div className="readout">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RangeControl({ label, min, max, step, value, onChange }) {
  return (
    <label className="rangeLabel">
      {label}
      <input min={min} max={max} step={step} type="range" value={value} onChange={(event) => onChange(event.target.value)} />
      <span>{Number(value).toFixed(step < 0.1 ? 2 : 1)}</span>
    </label>
  );
}

function EventList({ events, empty }) {
  if (!events.length) return <div className="emptyState compact">{empty}</div>;

  return (
    <div className="eventList">
      {events
        .slice()
        .reverse()
        .map((event, index) => (
          <div className="eventRow" key={`${event.t}-${index}`}>
            <code>{event.t}</code>
            <span>{event.type ?? "contact"}</span>
            <small>{event.objectId ?? event.a ?? ""}</small>
          </div>
        ))}
    </div>
  );
}

function applyImpulse(body, point, mode) {
  const raw = mode === "push" ? Vector.sub(body.position, point) : Vector.sub(point, body.position);
  const direction = Vector.magnitude(raw) < 1 ? { x: 1, y: -0.25 } : Vector.normalise(raw);
  const scale = mode === "push" ? 0.045 : 0.038;
  Body.applyForce(body, body.position, {
    x: direction.x * scale * body.mass,
    y: direction.y * scale * body.mass,
  });
}

function drawWorld(canvas, engine, worldSpec, viewState) {
  const ctx = canvas.getContext("2d");
  const { width, height } = worldSpec;
  ctx.clearRect(0, 0, width, height);
  drawBackground(ctx, worldSpec, viewState);
  drawTarget(ctx, worldSpec.target);
  drawSensorOverlay(ctx, worldSpec, viewState.observation, viewState.layers);
  drawConstraints(ctx, engine);
  drawBodies(ctx, engine, viewState.selectedBody, viewState.layers);
  drawPointer(ctx, viewState.pointer);
  drawCanvasHud(ctx, worldSpec, viewState);
}

function drawBackground(ctx, worldSpec, viewState) {
  const { width, height, light } = worldSpec;
  const gradient = ctx.createLinearGradient(
    width * (0.5 - Math.cos(light.angle) * 0.5),
    0,
    width * (0.5 + Math.cos(light.angle) * 0.5),
    height,
  );
  gradient.addColorStop(0, `rgba(255, 255, 255, ${0.86 + light.intensity * 0.08})`);
  gradient.addColorStop(1, "rgba(229, 235, 240, 0.94)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = viewState.layers.randomization ? "rgba(95, 112, 128, 0.13)" : "rgba(95, 112, 128, 0.08)";
  ctx.lineWidth = 1;

  for (let x = 0; x <= width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawTarget(ctx, target) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(47, 167, 143, 0.12)";
  ctx.strokeStyle = "#2fa78f";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#13695a";
  ctx.font = "600 11px Inter, system-ui";
  ctx.fillText("reward zone", target.x - 31, target.y + 4);
  ctx.restore();
}

function drawSensorOverlay(ctx, worldSpec, observation, layers) {
  if (!layers.sensors || !observation) return;
  const camera = worldSpec.camera;
  ctx.save();
  ctx.globalAlpha = 0.72;
  ctx.strokeStyle = "#0f766e";
  ctx.lineWidth = 1.2;
  observation.camera.rays.forEach((ray) => {
    ctx.beginPath();
    ctx.moveTo(camera.x, camera.y);
    ctx.lineTo(ray.end.x, ray.end.y);
    ctx.strokeStyle = ray.hitId ? "rgba(15, 118, 110, 0.48)" : "rgba(15, 118, 110, 0.18)";
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#0f766e";
  ctx.beginPath();
  ctx.arc(camera.x, camera.y, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 10px Inter, system-ui";
  ctx.fillText("C", camera.x - 3, camera.y + 4);
  ctx.restore();
}

function drawConstraints(ctx, engine) {
  ctx.save();
  ctx.lineWidth = 2;
  engine.world.constraints.forEach((constraint) => {
    const pointA = constraint.bodyA
      ? Vector.add(constraint.bodyA.position, constraint.pointA)
      : constraint.pointA;
    const pointB = constraint.bodyB
      ? Vector.add(constraint.bodyB.position, constraint.pointB)
      : constraint.pointB;
    if (!pointA || !pointB) return;
    ctx.beginPath();
    ctx.moveTo(pointA.x, pointA.y);
    ctx.lineTo(pointB.x, pointB.y);
    ctx.strokeStyle = constraint.label?.includes("soft") ? "rgba(201, 111, 84, 0.55)" : "rgba(51, 65, 85, 0.48)";
    ctx.stroke();
  });
  ctx.restore();
}

function drawBodies(ctx, engine, selectedBody, layers) {
  engine.world.bodies.forEach((body) => {
    const vertices = body.vertices;
    const fill = body.render?.fillStyle || "#a5b4c2";
    const stroke = body.render?.strokeStyle || "#55606d";
    const isSelected = selectedBody?.id === body.id;
    const isParticle = body.plugin.kind === "particle" || body.plugin.kind === "soft-node";

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(vertices[0].x, vertices[0].y);
    for (let index = 1; index < vertices.length; index += 1) {
      ctx.lineTo(vertices[index].x, vertices[index].y);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.strokeStyle = isSelected ? "#e29535" : stroke;
    ctx.lineWidth = isSelected ? 3 : body.isStatic ? 1.5 : 1.25;
    ctx.shadowColor = layers.randomization ? "rgba(36, 48, 63, 0.18)" : "transparent";
    ctx.shadowBlur = body.isStatic ? 0 : 8;
    ctx.shadowOffsetY = body.isStatic ? 0 : 4;
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.stroke();

    if (!body.isStatic && !isParticle && body.plugin.worldId) {
      ctx.fillStyle = "rgba(17, 24, 39, 0.72)";
      ctx.font = "600 9px Inter, system-ui";
      ctx.fillText(body.plugin.worldId.replace("body-", "b-"), body.position.x - 18, body.position.y + 3);
    }

    if (body.plugin.breakable && !body.plugin.fractured) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(body.position.x - 38, body.position.y - 10);
      ctx.lineTo(body.position.x + 32, body.position.y + 10);
      ctx.moveTo(body.position.x - 4, body.position.y - 14);
      ctx.lineTo(body.position.x + 24, body.position.y + 12);
      ctx.stroke();
    }

    ctx.restore();
  });
}

function drawPointer(ctx, pointer) {
  if (!pointer) return;
  ctx.save();
  ctx.beginPath();
  ctx.arc(pointer.x, pointer.y, 11, 0, Math.PI * 2);
  ctx.strokeStyle = "#e29535";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawCanvasHud(ctx, worldSpec, viewState) {
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
  ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
  roundedRect(ctx, 18, 16, 376, 58, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#111827";
  ctx.font = "700 13px Inter, system-ui";
  ctx.fillText("bounded Newtonian sandbox", 34, 39);
  ctx.font = "500 11px Inter, system-ui";
  ctx.fillStyle = "#55606d";
  ctx.fillText(
    `${worldSpec.width}x${worldSpec.height} px · dt 16.7 ms · seed ${viewState.seed}`,
    34,
    58,
  );
  ctx.fillStyle = viewState.recording ? "#0f766e" : "#9ca3af";
  ctx.beginPath();
  ctx.arc(367, 38, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function roundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function roundPoint(point) {
  return {
    x: Number(point.x.toFixed(2)),
    y: Number(point.y.toFixed(2)),
  };
}

export default App;
