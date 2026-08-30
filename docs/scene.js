/* LatentMesh — WebGL mesh-network scene.
   One persistent scene that changes state as the reader scrolls, rather than
   several separate canvases. Falls back silently to the SVG panels if WebGL
   is unavailable, and renders a single static frame under reduced-motion. */

import * as THREE from "three";

const CYAN = 0x50e6ff, GREEN = 0x4df3a5, VIOLET = 0x8d7cff, FAINT = 0x243951, DIM = 0x526379;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function createScene(canvas) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch (e) {
    return null; // caller falls back to SVG
  }
  if (!renderer.getContext()) return null;

  renderer.setClearColor(0x000000, 0);
  // Cap DPR: a phone at 3x costs 9x the fragments for no visible gain here.
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x050810, 12, 34);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 7.4, 13.5);
  camera.lookAt(0, 0.4, 0);

  const root = new THREE.Group();
  scene.add(root);

  /* ---------- terrain: a low wireframe ridge the nodes sit on ---------- */
  const W = 26, SEG = 40;
  const terrainGeo = new THREE.PlaneGeometry(W, W * 0.62, SEG, Math.round(SEG * 0.62));
  const pos = terrainGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    // Two offset sine ridges + a gentle bowl. Deterministic, no RNG.
    const h = Math.sin(x * 0.34) * 0.55 + Math.cos(y * 0.42) * 0.42 + Math.sin((x + y) * 0.18) * 0.3;
    pos.setZ(i, h - (x * x + y * y) * 0.004);
  }
  terrainGeo.computeVertexNormals();
  const terrain = new THREE.LineSegments(
    new THREE.WireframeGeometry(terrainGeo),
    new THREE.LineBasicMaterial({ color: FAINT, transparent: true, opacity: 0.34 })
  );
  terrain.rotation.x = -Math.PI / 2;
  terrain.position.y = -1.2;
  root.add(terrain);

  const groundHeight = (x, z) => {
    const y = -z; // plane was rotated
    return Math.sin(x * 0.34) * 0.55 + Math.cos(y * 0.42) * 0.42 + Math.sin((x + y) * 0.18) * 0.3
      - (x * x + y * y) * 0.004 - 1.2;
  };

  /* ---------- nodes ---------- */
  const NODE_XZ = [
    [-7.2, 2.6], [-2.4, -1.4], [2.6, 2.2], [7.0, -0.6], [-0.4, 4.6], [5.2, -3.8],
  ];
  const nodes = NODE_XZ.map(([x, z], i) => {
    const g = new THREE.Group();
    g.position.set(x, groundHeight(x, z) + 0.55, z);

    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.46, 0.46, 0.46),
      new THREE.MeshBasicMaterial({ color: CYAN, wireframe: true, transparent: true, opacity: 0.85 })
    );
    g.add(body);

    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.13, 12, 12),
      new THREE.MeshBasicMaterial({ color: CYAN })
    );
    g.add(core);

    // Mast so the nodes read as radios rather than floating cubes.
    const mast = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.23, 0), new THREE.Vector3(0, 0.78, 0)]),
      new THREE.LineBasicMaterial({ color: DIM, transparent: true, opacity: 0.7 })
    );
    g.add(mast);

    // Expanding RF ring, one per node, phase-offset.
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.3, 0.34, 40),
      new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: 0, side: THREE.DoubleSide })
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -0.3;
    g.add(ring);

    root.add(g);
    return { g, body, core, ring, phase: i * 0.62, isGateway: i === 3, x, z };
  });

  /* ---------- links ---------- */
  const LINKS = [[0, 1], [1, 2], [2, 3], [1, 4], [2, 5], [4, 2], [5, 3]];
  const links = LINKS.map(([a, b]) => {
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([nodes[a].g.position, nodes[b].g.position]),
      new THREE.LineBasicMaterial({ color: CYAN, transparent: true, opacity: 0 })
    );
    root.add(line);
    return { line, a, b };
  });

  /* ---------- packets travelling the links ---------- */
  const packets = LINKS.map(([a, b], i) => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.085, 10, 10),
      new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0 })
    );
    root.add(m);
    return { m, a, b, t: i * 0.17 };
  });

  /* ---------- uplink beam from the gateway ---------- */
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.5, 7, 16, 1, true),
    new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0, side: THREE.DoubleSide })
  );
  const gw = nodes[3];
  beam.position.set(gw.x, gw.g.position.y + 3.5, gw.z);
  root.add(beam);

  const cloud = new THREE.Mesh(
    new THREE.TorusGeometry(1.15, 0.045, 8, 44),
    new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0 })
  );
  cloud.position.set(gw.x, gw.g.position.y + 7.2, gw.z);
  cloud.rotation.x = Math.PI / 2;
  root.add(cloud);

  /* ---------- state ---------- */
  const S = {
    v1: { links: 0, packets: 0, rings: 0, beam: 0, node: DIM, spin: 0.05, cam: [0, 7.4, 13.5] },
    v2: { links: 0.55, packets: 1, rings: 1, beam: 0, node: CYAN, spin: 0.11, cam: [0, 6.2, 12.0] },
    v4: { links: 0.28, packets: 0.35, rings: 0.4, beam: 0, node: CYAN, spin: 0.07, cam: [-2, 5.4, 11.4] },
    v6: { links: 0.5, packets: 0.9, rings: 0.7, beam: 1, node: CYAN, spin: 0.09, cam: [3.4, 6.6, 11.8] },
  };
  let target = S.v1, cur = { links: 0, packets: 0, rings: 0, beam: 0, spin: 0.05 };
  const camTarget = new THREE.Vector3(...S.v1.cam);

  const setState = (id) => { if (S[id]) { target = S[id]; camTarget.set(...S[id].cam); } };

  /* ---------- resize ---------- */
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  addEventListener("resize", resize, { passive: true });

  /* ---------- loop ---------- */
  let running = true, t = 0;
  const lerp = (a, b, k) => a + (b - a) * k;

  const frame = (ms) => {
    if (!running) return;
    requestAnimationFrame(frame);
    const time = ms * 0.001;
    const dt = Math.min(0.05, time - t || 0.016);
    t = time;

    cur.links = lerp(cur.links, target.links, 0.05);
    cur.packets = lerp(cur.packets, target.packets, 0.05);
    cur.rings = lerp(cur.rings, target.rings, 0.05);
    cur.beam = lerp(cur.beam, target.beam, 0.05);
    cur.spin = lerp(cur.spin, target.spin, 0.03);

    root.rotation.y += dt * cur.spin;
    camera.position.lerp(camTarget, 0.02);
    camera.lookAt(0, 0.4, 0);

    nodes.forEach((n, i) => {
      n.body.rotation.y += dt * 0.5;
      n.body.rotation.x += dt * 0.22;
      n.core.material.color.setHex(target.node === DIM && !n.isGateway ? DIM : (n.isGateway && cur.beam > 0.4 ? GREEN : CYAN));
      n.body.material.color.copy(n.core.material.color);
      // node 5 is the "dark" node in the store-and-forward state
      const dark = target === S.v4 && i === 5;
      n.g.visible = true;
      n.core.material.opacity = dark ? 0.25 + Math.sin(time * 3) * 0.15 : 1;
      n.core.material.transparent = true;

      const p = ((time * 0.42 + n.phase) % 1);
      n.ring.scale.setScalar(1 + p * 6);
      n.ring.material.opacity = cur.rings * (1 - p) * 0.5 * (dark ? 0.2 : 1);
    });

    links.forEach((l, i) => {
      const dark = target === S.v4 && (l.a === 5 || l.b === 5);
      l.line.material.opacity = cur.links * (dark ? 0.12 : 1);
    });

    packets.forEach((p) => {
      p.t = (p.t + dt * 0.32) % 1;
      const a = nodes[p.a].g.position, b = nodes[p.b].g.position;
      p.m.position.lerpVectors(a, b, p.t);
      p.m.position.y += Math.sin(p.t * Math.PI) * 0.5; // arc the hop
      const dark = target === S.v4 && (p.a === 5 || p.b === 5);
      p.m.material.opacity = cur.packets * (dark ? 0 : 1) * (0.35 + Math.sin(p.t * Math.PI) * 0.65);
    });

    beam.material.opacity = cur.beam * 0.17;
    beam.rotation.y += dt * 0.6;
    cloud.material.opacity = cur.beam * 0.75;
    cloud.rotation.z += dt * 0.35;
    cloud.scale.setScalar(1 + Math.sin(time * 1.4) * 0.03);

    renderer.render(scene, camera);
  };

  if (reduce) {
    // One static, fully-formed frame. No loop, no motion.
    cur = { links: 0.5, packets: 0.8, rings: 0.35, beam: 0.6, spin: 0 };
    target = S.v2;
    links.forEach((l) => (l.line.material.opacity = 0.5));
    packets.forEach((p) => (p.m.material.opacity = 0.8));
    nodes.forEach((n) => (n.ring.material.opacity = 0.18));
    renderer.render(scene, camera);
  } else {
    requestAnimationFrame(frame);
  }

  // Pause entirely when the canvas is off-screen — no GPU burn while reading
  // the rest of the page.
  if (!reduce && "IntersectionObserver" in window) {
    new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (e.isIntersecting && !running) { running = true; t = 0; requestAnimationFrame(frame); }
        else if (!e.isIntersecting) running = false;
      });
    }, { threshold: 0.01 }).observe(canvas);
  }

  return { setState, resize };
}

/* ---------------------------------------------------------------------------
   Hero scene — an ambient node sphere. Distinct from the terrain scene above:
   this one reads as "cognition distributed through an environment" rather than
   "radios on a hillside", which is what the hero copy is about.
   --------------------------------------------------------------------------- */
export function createHeroScene(canvas) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch (e) { return null; }
  if (!renderer.getContext()) return null;

  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 60);
  camera.position.set(0, 0, 12.2);

  const root = new THREE.Group();
  root.rotation.z = 0.24;
  scene.add(root);

  // Fibonacci sphere: even coverage without clustering at the poles.
  const N = 46, R = 4.1;
  const pts = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = golden * i;
    pts.push(new THREE.Vector3(Math.cos(th) * r * R, y * R, Math.sin(th) * r * R));
  }

  // Nodes
  const nodeGeo = new THREE.SphereGeometry(0.062, 8, 8);
  const nodeMat = new THREE.MeshBasicMaterial({ color: CYAN, transparent: true, opacity: 0.92 });
  const inst = new THREE.InstancedMesh(nodeGeo, nodeMat, N);
  const dummy = new THREE.Object3D();
  pts.forEach((p, i) => { dummy.position.copy(p); dummy.updateMatrix(); inst.setMatrixAt(i, dummy.matrix); });
  inst.instanceMatrix.needsUpdate = true;
  root.add(inst);

  // Links between near neighbours only — a full graph reads as noise.
  const segs = [], pairs = [];
  const LINK_MAX = R * 1.02;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      if (pts[i].distanceTo(pts[j]) < LINK_MAX) { segs.push(pts[i], pts[j]); pairs.push([i, j]); }
    }
  }
  const lines = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(segs),
    new THREE.LineBasicMaterial({ color: CYAN, transparent: true, opacity: 0.17 })
  );
  root.add(lines);

  // A halo so the sphere reads as volume rather than a flat scatter.
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(R * 1.16, 30, 30),
    new THREE.MeshBasicMaterial({ color: VIOLET, transparent: true, opacity: 0.045, side: THREE.BackSide })
  );
  root.add(halo);

  // Packets hopping real edges, so the motion matches the topology.
  const PK = 14;
  const pkGeo = new THREE.SphereGeometry(0.075, 8, 8);
  const pkMat = new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.95 });
  const pk = new THREE.InstancedMesh(pkGeo, pkMat, PK);
  root.add(pk);
  const flights = Array.from({ length: PK }, (_, i) => ({
    e: (i * 7) % pairs.length, t: (i / PK), speed: 0.24 + (i % 5) * 0.045,
  }));

  const resize = () => {
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, r.width), h = Math.max(1, r.height);
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  resize();
  addEventListener("resize", resize, { passive: true });

  // Gentle parallax toward the pointer — desktop only, and never on touch.
  let px = 0, py = 0, tx = 0, ty = 0;
  if (matchMedia("(hover: hover) and (pointer: fine)").matches) {
    addEventListener("pointermove", (e) => {
      tx = (e.clientX / innerWidth - 0.5) * 0.22;
      ty = (e.clientY / innerHeight - 0.5) * 0.16;
    }, { passive: true });
  }

  let running = true, last = 0;
  const frame = (ms) => {
    if (!running) return;
    requestAnimationFrame(frame);
    const t = ms * 0.001, dt = Math.min(0.05, t - last || 0.016);
    last = t;

    root.rotation.y += dt * 0.085;
    px += (tx - px) * 0.045; py += (ty - py) * 0.045;
    root.rotation.x = -py + Math.sin(t * 0.24) * 0.05;
    camera.position.x = px * 3.4;
    camera.lookAt(0, 0, 0);

    flights.forEach((f, i) => {
      f.t += dt * f.speed;
      if (f.t >= 1) { f.t = 0; f.e = (f.e + 11) % pairs.length; }
      const [a, b] = pairs[f.e];
      dummy.position.lerpVectors(pts[a], pts[b], f.t).multiplyScalar(1.012);
      const s = 0.55 + Math.sin(f.t * Math.PI) * 0.75;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      pk.setMatrixAt(i, dummy.matrix);
    });
    pk.instanceMatrix.needsUpdate = true;

    lines.material.opacity = 0.14 + Math.sin(t * 0.7) * 0.035;
    renderer.render(scene, camera);
  };

  if (reduce) {
    flights.forEach((f, i) => {
      const [a, b] = pairs[f.e];
      dummy.position.lerpVectors(pts[a], pts[b], f.t).multiplyScalar(1.012);
      dummy.scale.setScalar(1); dummy.updateMatrix(); pk.setMatrixAt(i, dummy.matrix);
    });
    pk.instanceMatrix.needsUpdate = true;
    renderer.render(scene, camera);
  } else {
    requestAnimationFrame(frame);
  }

  if (!reduce && "IntersectionObserver" in window) {
    new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (e.isIntersecting && !running) { running = true; last = 0; requestAnimationFrame(frame); }
        else if (!e.isIntersecting) running = false;
      });
    }, { threshold: 0.01 }).observe(canvas);
  }

  return { resize };
}
