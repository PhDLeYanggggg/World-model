export function hashSeed(input) {
  const text = String(input || "world-lab");
  let hash = 2166136261;

  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

export function createRng(seed) {
  let state = hashSeed(seed);

  return function next() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function range(rng, min, max) {
  return min + (max - min) * rng();
}

export function pick(rng, values) {
  return values[Math.floor(rng() * values.length) % values.length];
}
