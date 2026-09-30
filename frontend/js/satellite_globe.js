// ============================================================
// Satellite Intelligence & Global Fraud Defense System
// 1:1 Fuselab Creative Starlink Intelligence Interface Engine
// 3D Orbital Earth, Multi-Plane Constellation Mesh & Laser Strike
// ============================================================

(function () {
    "use strict";

    // ============================================================
    // Strict Project Locations (California, Texas, New York, Florida)
    // Directly mapped from final_model training features & 197 batch
    // ============================================================
    const PROJECT_LOCATIONS = {
        "california": {
            lat: 36.7783,
            lon: -119.4179,
            city: "California, USA",
            hub: "Silicon Valley Tech & Banking Hub",
            flag: "🇺🇸",
            ipPrefix: "240.44",
            tag: "LOC-CA-01",
            load: 95,
            devices: 1884
        },
        "texas": {
            lat: 31.9686,
            lon: -99.9018,
            city: "Texas, USA",
            hub: "Dallas / Austin Financial District",
            flag: "🇺🇸",
            ipPrefix: "28.75",
            tag: "LOC-TX-02",
            load: 81,
            devices: 1738
        },
        "new york": {
            lat: 40.7128,
            lon: -74.0060,
            city: "New York, USA",
            hub: "Wall Street Clearing Center",
            flag: "🇺🇸",
            ipPrefix: "198.51",
            tag: "LOC-NY-03",
            load: 72,
            devices: 2140
        },
        "florida": {
            lat: 27.6648,
            lon: -81.5158,
            city: "Florida, USA",
            hub: "Miami Interbank Corridor",
            flag: "🇺🇸",
            ipPrefix: "172.56",
            tag: "LOC-FL-04",
            load: 45,
            devices: 1258
        }
    };

    // Core Constants
    const GLOBE_RADIUS = 6.2;
    const SATELLITE_ORBIT_RADIUS = 9.8;

    // Three.js Core Objects
    let scene, camera, renderer, container;
    let earthMesh, atmosphereMesh;
    let constellationGroup, constellationSatellites = [], orbitalRings = [];
    let interSatLinkLines;
    let activeSatelliteGroup, activeSatSolarWing, activeSatStrobe, activeSatBeaconLight, activeSatFlareSprite, activeSatPlumes = [];
    let laserBeamMesh, downlinkAuraMesh, laserSparks = [];
    let uplinkBeamMesh, uplinkAuraMesh, uplinkPackets = [];
    let groundLaunchGroup, launchRings = [];
    let satelliteNeuralHalo, satelliteComputeRings = [], activeSatComputeLight;
    let groundPingGroup, pingRings = [];
    let groundMarkerGroup;
    let activeInterception = null;
    let activeMovieSequence = null; // Master 3-Phase Cinematic State

    // Interaction & Animation State
    let isUserDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let globeRotationSpeed = 0.0016;
    let cameraMode = "orbital"; // "orbital", "chase", "target"
    let targetCameraPos = null;
    let targetCameraLook = null;
    let currentOrbitAngle = 0.85;

    // ============================================================
    // Cinematic Movie Web Audio Synthesizer (Uplink -> Inference -> Downlink)
    // ============================================================
    let audioCtx = null;
    function playMovieAudio(phase, isFraud) {
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === "suspended") audioCtx.resume();
            const now = audioCtx.currentTime;

            if (phase === "uplink") {
                // Scene 1: Ground-to-Space Doppler Uplink Sweep
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(220, now);
                osc.frequency.exponentialRampToValueAtTime(1450, now + 0.85);
                gain.gain.setValueAtTime(0.01, now);
                gain.gain.linearRampToValueAtTime(0.14, now + 0.2);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.85);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(now);
                osc.stop(now + 0.85);

                // Digital data chirp bursts (Ku-Band telemetry packets)
                [0.1, 0.25, 0.42, 0.6].forEach((delay, idx) => {
                    const chirp = audioCtx.createOscillator();
                    const cGain = audioCtx.createGain();
                    chirp.type = "sine";
                    chirp.frequency.setValueAtTime(1400 + idx * 280, now + delay);
                    cGain.gain.setValueAtTime(0.07, now + delay);
                    cGain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.08);
                    chirp.connect(cGain);
                    cGain.connect(audioCtx.destination);
                    chirp.start(now + delay);
                    chirp.stop(now + delay + 0.08);
                });

            } else if (phase === "inference") {
                // Scene 2: Deep Orbital Neural Computing Hum + AI Resonance
                const subOsc = audioCtx.createOscillator();
                const subGain = audioCtx.createGain();
                subOsc.type = "triangle";
                subOsc.frequency.setValueAtTime(85, now);
                subOsc.frequency.linearRampToValueAtTime(120, now + 0.5);
                subOsc.frequency.linearRampToValueAtTime(95, now + 1.0);
                subGain.gain.setValueAtTime(0.02, now);
                subGain.gain.linearRampToValueAtTime(0.12, now + 0.3);
                subGain.gain.exponentialRampToValueAtTime(0.001, now + 1.1);
                subOsc.connect(subGain);
                subGain.connect(audioCtx.destination);
                subOsc.start(now);
                subOsc.stop(now + 1.1);

                // Processing Harmonic Arpeggio
                [1760, 2200, 2640, 3520].forEach((freq, i) => {
                    const tick = audioCtx.createOscillator();
                    const tGain = audioCtx.createGain();
                    tick.type = "sine";
                    tick.frequency.setValueAtTime(freq, now + 0.2 + i * 0.12);
                    tGain.gain.setValueAtTime(0.04, now + 0.2 + i * 0.12);
                    tGain.gain.exponentialRampToValueAtTime(0.001, now + 0.2 + i * 0.12 + 0.07);
                    tick.connect(tGain);
                    tGain.connect(audioCtx.destination);
                    tick.start(now + 0.2 + i * 0.12);
                    tick.stop(now + 0.2 + i * 0.12 + 0.07);
                });

            } else if (phase === "downlink") {
                // Scene 3: Satellite Return Laser Downlink
                if (isFraud) {
                    // Massive Sci-Fi Laser Discharge + Ground Zero Shockwave Boom
                    const laserOsc = audioCtx.createOscillator();
                    const laserGain = audioCtx.createGain();
                    laserOsc.type = "sawtooth";
                    laserOsc.frequency.setValueAtTime(1850, now);
                    laserOsc.frequency.exponentialRampToValueAtTime(90, now + 0.45);
                    laserGain.gain.setValueAtTime(0.22, now);
                    laserGain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
                    laserOsc.connect(laserGain);
                    laserGain.connect(audioCtx.destination);
                    laserOsc.start(now);
                    laserOsc.stop(now + 0.45);

                    // Low-Frequency Ground Impact Boom (Sub-bass rumble)
                    const boomOsc = audioCtx.createOscillator();
                    const boomGain = audioCtx.createGain();
                    boomOsc.type = "sine";
                    boomOsc.frequency.setValueAtTime(58, now + 0.12);
                    boomOsc.frequency.exponentialRampToValueAtTime(26, now + 0.9);
                    boomGain.gain.setValueAtTime(0.28, now + 0.12);
                    boomGain.gain.exponentialRampToValueAtTime(0.001, now + 0.9);
                    boomOsc.connect(boomGain);
                    boomGain.connect(audioCtx.destination);
                    boomOsc.start(now + 0.12);
                    boomOsc.stop(now + 0.9);

                    // Urgent Dual Interception Alarm Warble
                    [0.4, 0.65].forEach((d) => {
                        const warn = audioCtx.createOscillator();
                        const wGain = audioCtx.createGain();
                        warn.type = "square";
                        warn.frequency.setValueAtTime(880, now + d);
                        warn.frequency.setValueAtTime(740, now + d + 0.09);
                        wGain.gain.setValueAtTime(0.06, now + d);
                        wGain.gain.exponentialRampToValueAtTime(0.001, now + d + 0.18);
                        warn.connect(wGain);
                        wGain.connect(audioCtx.destination);
                        warn.start(now + d);
                        warn.stop(now + d + 0.18);
                    });
                } else {
                    // Harmonious Emerald Clearance Chime (C-Maj Chord)
                    [523.25, 659.25, 783.99, 1046.5].forEach((freq, idx) => {
                        const note = audioCtx.createOscillator();
                        const nGain = audioCtx.createGain();
                        note.type = "sine";
                        note.frequency.setValueAtTime(freq, now + idx * 0.08);
                        nGain.gain.setValueAtTime(0.09, now + idx * 0.08);
                        nGain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.08 + 0.6);
                        note.connect(nGain);
                        nGain.connect(audioCtx.destination);
                        note.start(now + idx * 0.08);
                        note.stop(now + idx * 0.08 + 0.6);
                    });
                }
            }
        } catch (e) {
            // Audio policy silently handled
        }
    }

    // Alias for backward compatibility
    function playLaserAudio(isFraud) {
        playMovieAudio("downlink", isFraud);
    }

    // Convert Lat/Lon to 3D Cartesian coordinates
    function latLonToVector3(lat, lon, radius) {
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);
        const x = -(radius * Math.sin(phi) * Math.cos(theta));
        const z = radius * Math.sin(phi) * Math.sin(theta);
        const y = radius * Math.cos(phi);
        return new THREE.Vector3(x, y, z);
    }

    function resolveLocation(locStr) {
        if (!locStr) return PROJECT_LOCATIONS["california"];
        const lower = String(locStr).toLowerCase().trim();
        if (lower.includes("california") || lower.includes("ca")) return PROJECT_LOCATIONS["california"];
        if (lower.includes("texas") || lower.includes("tx")) return PROJECT_LOCATIONS["texas"];
        if (lower.includes("new york") || lower.includes("ny")) return PROJECT_LOCATIONS["new york"];
        if (lower.includes("florida") || lower.includes("fl")) return PROJECT_LOCATIONS["florida"];

        const keys = ["california", "texas", "new york", "florida"];
        let hash = 0;
        for (let i = 0; i < lower.length; i++) hash = (hash * 31 + lower.charCodeAt(i)) & 0xffffffff;
        return PROJECT_LOCATIONS[keys[Math.abs(hash) % keys.length]];
    }

    // ============================================================
    // Initialize 3D Theater
    // ============================================================
    function initSatelliteTheater() {
        container = document.getElementById("satelliteTheaterContainer");
        if (!container || typeof THREE === "undefined") return;

        const width = container.clientWidth || 920;
        const height = container.clientHeight || 560;

        // 1. Scene & Camera
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
        camera.position.set(0, 4.5, 23);

        // 2. WebGL Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.35;

        // Clean existing
        container.querySelectorAll("canvas.three-canvas").forEach((c) => c.remove());
        renderer.domElement.className = "three-canvas";
        renderer.domElement.style.width = "100%";
        renderer.domElement.style.height = "100%";
        renderer.domElement.style.display = "block";
        container.appendChild(renderer.domElement);

        // 3. Cinematic Lighting (Sunlight + Ambient Earthshine + Rim Lighting)
        const ambientLight = new THREE.AmbientLight(0x0e1726, 1.8);
        scene.add(ambientLight);

        // Sunlight key light from upper right (creates Earth terminator crescent)
        const sunLight = new THREE.DirectionalLight(0xfff5e4, 2.8);
        sunLight.position.set(25, 14, 16);
        scene.add(sunLight);

        // Electric cyan rim light (atmospheric backlight)
        const cyanRim = new THREE.DirectionalLight(0x38bdf8, 1.4);
        cyanRim.position.set(-22, -8, -18);
        scene.add(cyanRim);

        // Deep blue filler
        const blueFill = new THREE.PointLight(0x1e3a8a, 1.2, 50);
        blueFill.position.set(0, -15, 10);
        scene.add(blueFill);

        // 4. Construct Photorealistic Earth
        buildEarthGlobe();

        // 5. Construct Mega-Constellation Orbital Mesh (Image 4 Starlink Lattice)
        buildConstellationMesh();

        // 6. Construct Primary Active Starlink Satellite
        buildActiveStarlinkSatellite();

        // 7. Construct Laser Targeting System
        buildLaserTargetingSystem();

        // 8. Construct Persistent Ground Markers (Image 4 Red Reticle & Washington Tag)
        buildGroundMarkers();

        // 9. Attach Interactions & Controls
        initInteractions();

        // 10. Start 60 FPS Render Loop
        animate();
    }

    // ============================================================
    // 3D Earth Globe with Night City Lights & Atmospheric Corona
    // ============================================================
    function buildEarthGlobe() {
        const textureLoader = new THREE.TextureLoader();

        const nightTexture = textureLoader.load(
            "assets/textures/earth_night.png",
            () => renderer && renderer.render(scene, camera),
            undefined,
            () => console.warn("Fallback to procedural earth")
        );

        const dayTexture = textureLoader.load(
            "assets/textures/earth_day.jpg",
            () => renderer && renderer.render(scene, camera)
        );

        // Earth Spherical Mesh
        const earthMaterial = new THREE.MeshPhongMaterial({
            map: dayTexture,
            emissiveMap: nightTexture,
            emissive: new THREE.Color(0xffd560),
            emissiveIntensity: 1.5,
            specular: new THREE.Color(0x38bdf8),
            shininess: 38,
            bumpScale: 0.04,
        });

        const earthGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
        earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
        // Tilt Earth axis ~23.5 degrees like real Earth & orient North America front-facing
        earthMesh.rotation.z = 0.22;
        earthMesh.rotation.y = 1.35;
        scene.add(earthMesh);

        // Atmospheric Rayleigh Scattering Outer Shell (Image 4 Cyan Rim Halo)
        const atmosGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.05, 64, 64);
        const atmosMaterial = new THREE.ShaderMaterial({
            vertexShader: `
                varying vec3 vNormal;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                varying vec3 vNormal;
                void main() {
                    float intensity = pow(0.72 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.5);
                    gl_FragColor = vec4(0.22, 0.74, 0.97, 1.0) * intensity * 1.6;
                }
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true,
        });
        atmosphereMesh = new THREE.Mesh(atmosGeometry, atmosMaterial);
        scene.add(atmosphereMesh);

        // Secondary subtle inner atmosphere haze
        const innerAtmosGeo = new THREE.SphereGeometry(GLOBE_RADIUS * 1.015, 48, 48);
        const innerAtmosMat = new THREE.MeshBasicMaterial({
            color: 0x0284c7,
            transparent: true,
            opacity: 0.12,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
        });
        const innerAtmos = new THREE.Mesh(innerAtmosGeo, innerAtmosMat);
        scene.add(innerAtmos);
    }

    // ============================================================
    // Mega-Constellation Orbital Mesh (Image 4 Starlink Shell)
    // Multiple Inclined Orbital Rings + Inter-Satellite Laser Links
    // ============================================================
    function buildConstellationMesh() {
        constellationGroup = new THREE.Group();
        scene.add(constellationGroup);

        // 8 Real-World Orbital Inclination Shells (like Starlink Gen2 LEO)
        const ORBITAL_PLANES = [
            { radius: 9.2, inclX: 0.92, inclZ: 0.15, rotY: 0.0, color: 0x38bdf8, sats: 6, speed: 0.0035 },
            { radius: 9.4, inclX: -0.85, inclZ: -0.22, rotY: 0.78, color: 0x0284c7, sats: 6, speed: 0.0038 },
            { radius: 9.0, inclX: 0.72, inclZ: 0.35, rotY: 1.57, color: 0x38bdf8, sats: 5, speed: 0.0032 },
            { radius: 9.6, inclX: -0.70, inclZ: 0.18, rotY: 2.35, color: 0x60a5fa, sats: 6, speed: 0.0036 },
            { radius: 9.1, inclX: 1.15, inclZ: -0.12, rotY: 3.14, color: 0x0284c7, sats: 5, speed: 0.0033 },
            { radius: 9.5, inclX: -1.05, inclZ: 0.28, rotY: 3.92, color: 0x38bdf8, sats: 6, speed: 0.0037 },
            { radius: 9.8, inclX: 0.45, inclZ: -0.32, rotY: 4.71, color: 0xd4af37, sats: 6, speed: 0.0034 },
            { radius: 9.3, inclX: -0.42, inclZ: 0.25, rotY: 5.50, color: 0x38bdf8, sats: 5, speed: 0.0039 },
        ];

        const satGeometry = new THREE.BoxGeometry(0.14, 0.14, 0.14);
        const satMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });

        const ringSegments = 96;

        ORBITAL_PLANES.forEach((plane, planeIdx) => {
            // 1. Orbital Trajectory Line Loop
            const points = [];
            for (let i = 0; i <= ringSegments; i++) {
                const theta = (i / ringSegments) * Math.PI * 2;
                const localX = Math.cos(theta) * plane.radius;
                const localZ = Math.sin(theta) * plane.radius;
                const v = new THREE.Vector3(localX, 0, localZ);
                v.applyEuler(new THREE.Euler(plane.inclX, plane.rotY, plane.inclZ));
                points.push(v);
            }

            const ringGeo = new THREE.BufferGeometry().setFromPoints(points);
            const ringMat = new THREE.LineBasicMaterial({
                color: plane.color,
                transparent: true,
                opacity: planeIdx === 6 ? 0.35 : 0.22,
                blending: THREE.AdditiveBlending,
            });
            const ringLine = new THREE.Line(ringGeo, ringMat);
            constellationGroup.add(ringLine);
            orbitalRings.push(ringLine);

            // 2. Populate Satellites Along This Orbit
            for (let s = 0; s < plane.sats; s++) {
                const satMesh = new THREE.Mesh(satGeometry, satMaterial);
                const satData = {
                    plane: plane,
                    angle: (s / plane.sats) * Math.PI * 2 + planeIdx * 0.4,
                    mesh: satMesh,
                    speed: plane.speed,
                    strobePhase: Math.random() * Math.PI * 2,
                };

                constellationGroup.add(satMesh);
                constellationSatellites.push(satData);
            }
        });

        // 3. Dynamic Inter-Satellite Laser Links (LineSegments)
        const maxLinks = 40;
        const linkPositions = new Float32Array(maxLinks * 2 * 3);
        const linkGeo = new THREE.BufferGeometry();
        linkGeo.setAttribute("position", new THREE.BufferAttribute(linkPositions, 3));
        const linkMat = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.22,
            blending: THREE.AdditiveBlending,
        });
        interSatLinkLines = new THREE.LineSegments(linkGeo, linkMat);
        scene.add(interSatLinkLines);
    }

    // ============================================================
    // Procedural PBR Canvas Texture Generators for Starlink Satellite
    // ============================================================

    // 1. High-Resolution Photovoltaic Silicon Solar Array (4 Folding Segments)
    function createStarlinkSolarTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 1024;
        canvas.height = 2048;
        const ctx = canvas.getContext("2d");

        // Deep titanium black frame
        ctx.fillStyle = "#070b14";
        ctx.fillRect(0, 0, 1024, 2048);

        const panelCount = 4;
        const panelH = (2048 - 60) / panelCount;
        const padX = 32;

        for (let p = 0; p < panelCount; p++) {
            const topY = 14 + p * (panelH + 10);
            const pWidth = 1024 - padX * 2;

            // Panel frame backing
            ctx.fillStyle = "#0c1322";
            ctx.fillRect(padX, topY, pWidth, panelH);
            ctx.strokeStyle = "#334155";
            ctx.lineWidth = 3;
            ctx.strokeRect(padX, topY, pWidth, panelH);

            // Photovoltaic Silicon Wafers Grid (8 columns x 22 rows per segment)
            const cols = 8;
            const rows = 22;
            const cellW = (pWidth - 20) / cols;
            const cellH = (panelH - 24) / rows;

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const cx = padX + 10 + c * cellW;
                    const cy = topY + 12 + r * cellH;

                    // Silicon Cell Gradient (Cobalt blue with anti-reflective sheen)
                    const grad = ctx.createLinearGradient(cx, cy, cx + cellW, cy + cellH);
                    const specShift = ((c * 7 + r * 13 + p * 19) % 5) * 5;
                    grad.addColorStop(0, `rgb(${14 + specShift}, ${38 + specShift}, ${115 + specShift})`);
                    grad.addColorStop(0.5, `rgb(${24 + specShift}, ${62 + specShift}, ${150 + specShift})`);
                    grad.addColorStop(1, `rgb(${12 + specShift}, ${32 + specShift}, ${95 + specShift})`);

                    ctx.fillStyle = grad;
                    ctx.fillRect(cx + 1, cy + 1, cellW - 2, cellH - 2);

                    // Silver Grid Fingers (Conductive micro-wires)
                    ctx.strokeStyle = "rgba(186, 230, 253, 0.4)";
                    ctx.lineWidth = 0.8;
                    for (let f = 1; f < 5; f++) {
                        const fy = cy + (f * cellH) / 5;
                        ctx.beginPath();
                        ctx.moveTo(cx + 2, fy);
                        ctx.lineTo(cx + cellW - 2, fy);
                        ctx.stroke();
                    }

                    // Dual Silver Busbars per column
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
                    ctx.lineWidth = 1.6;
                    const b1 = cx + cellW * 0.32;
                    const b2 = cx + cellW * 0.68;
                    ctx.beginPath();
                    ctx.moveTo(b1, cy);
                    ctx.lineTo(b1, cy + cellH);
                    ctx.moveTo(b2, cy);
                    ctx.lineTo(b2, cy + cellH);
                    ctx.stroke();
                }
            }

            // Mechanical Hinge Line & Gold Contacts between panels
            if (p < panelCount - 1) {
                const hy = topY + panelH + 5;
                ctx.fillStyle = "#1e293b";
                ctx.fillRect(padX - 8, hy - 4, pWidth + 16, 8);
                ctx.fillStyle = "#eab308"; // gold hinge pins
                for (let h = 0; h < 6; h++) {
                    const hx = padX + (h * pWidth) / 5;
                    ctx.beginPath();
                    ctx.arc(hx, hy, 3.5, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }

        // Mission Stencil at Base of Solar Wing
        ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
        ctx.font = "bold 18px monospace";
        ctx.fillText("STARLINK PV-ARRAY BATCH 345 // 100% PWR", 45, 2035);

        const tex = new THREE.CanvasTexture(canvas);
        tex.anisotropy = 8;
        return tex;
    }

    function createStarlinkSolarBumpTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 512;
        canvas.height = 1024;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, 512, 1024);

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        for (let i = 0; i < 4; i++) {
            ctx.strokeRect(16, 10 + i * 255, 480, 245);
        }
        for (let x = 30; x < 490; x += 30) {
            ctx.beginPath();
            ctx.moveTo(x, 10);
            ctx.lineTo(x, 1015);
            ctx.stroke();
        }
        return new THREE.CanvasTexture(canvas);
    }

    // 2. Chassis Top Plate Texture (Recessed Seams, Starlink Swoosh Logos, Mission Serials)
    function createStarlinkChassisTopTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext("2d");

        // Aerospace white/light grey composite body
        ctx.fillStyle = "#dce1e8";
        ctx.fillRect(0, 0, 1024, 512);

        // Subtle composite surface variations
        ctx.fillStyle = "rgba(255, 255, 255, 0.12)";
        for (let i = 0; i < 200; i++) {
            ctx.fillRect(Math.random() * 1024, Math.random() * 512, Math.random() * 12, Math.random() * 6);
        }

        // Recessed Modular Seams
        ctx.strokeStyle = "#475569";
        ctx.lineWidth = 3;
        ctx.strokeRect(30, 30, 964, 452);

        ctx.beginPath();
        ctx.moveTo(30, 256);
        ctx.lineTo(994, 256);
        ctx.stroke();

        const vSeams = [230, 430, 630, 830];
        vSeams.forEach((x) => {
            ctx.beginPath();
            ctx.moveTo(x, 30);
            ctx.lineTo(x, 482);
            ctx.stroke();
        });

        // Fastener / Rivet Dots
        ctx.fillStyle = "#334155";
        for (let x = 36; x < 994; x += 18) {
            ctx.fillRect(x, 32, 2, 2);
            ctx.fillRect(x, 254, 2, 2);
            ctx.fillRect(x, 480, 2, 2);
        }

        // Starlink Constellation Swoosh Logo 1 (Forward quadrant)
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.arc(320, 140, 50, -0.6, 1.4);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(320, 140, 70, -0.3, 1.1);
        ctx.stroke();

        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 20px monospace";
        ctx.fillText("OPERATOR", 275, 175);
        ctx.font = "14px monospace";
        ctx.fillText("LEO-X182", 280, 195);

        // Starlink Mission Marking 2 (Aft quadrant)
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.arc(720, 360, 50, -0.6, 1.4);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(720, 360, 70, -0.3, 1.1);
        ctx.stroke();

        ctx.fillStyle = "#0f172a";
        ctx.font = "bold 20px monospace";
        ctx.fillText("STARLINK LEO-345", 640, 420);
        ctx.font = "14px monospace";
        ctx.fillText("MISSION / DATA LINK", 640, 440);

        // Service Access Hatch with Hazard Striping
        ctx.fillStyle = "#94a3b8";
        ctx.fillRect(840, 75, 120, 95);
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 2;
        ctx.strokeRect(840, 75, 120, 95);

        ctx.fillStyle = "#eab308";
        for (let s = 0; s < 110; s += 20) {
            ctx.beginPath();
            ctx.moveTo(845 + s, 80);
            ctx.lineTo(860 + s, 80);
            ctx.lineTo(850 + s, 165);
            ctx.lineTo(835 + s, 165);
            ctx.fill();
        }

        // Gold Conduit Cable route
        ctx.strokeStyle = "#d97706";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(40, 490);
        ctx.lineTo(980, 490);
        ctx.stroke();

        const tex = new THREE.CanvasTexture(canvas);
        tex.anisotropy = 4;
        return tex;
    }

    // 3. Procedural Crinkled Gold MLI (Multi-Layer Insulation) Foil Blanket
    function createStarlinkGoldFoilTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext("2d");

        const grad = ctx.createLinearGradient(0, 0, 512, 512);
        grad.addColorStop(0, "#ca8a04");
        grad.addColorStop(0.3, "#eab308");
        grad.addColorStop(0.6, "#a16207");
        grad.addColorStop(0.85, "#fde047");
        grad.addColorStop(1, "#854d0e");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 512, 512);

        // Wrinkle highlights & shadow creases
        ctx.lineWidth = 1.5;
        for (let i = 0; i < 350; i++) {
            const x1 = Math.random() * 512;
            const y1 = Math.random() * 512;
            const x2 = x1 + (Math.random() - 0.5) * 60;
            const y2 = y1 + (Math.random() - 0.5) * 60;

            ctx.strokeStyle = "rgba(254, 240, 138, 0.7)";
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();

            ctx.strokeStyle = "rgba(69, 26, 3, 0.65)";
            ctx.beginPath();
            ctx.moveTo(x1 + 1.2, y1 + 1.2);
            ctx.lineTo(x2 + 1.2, y2 + 1.2);
            ctx.stroke();
        }
        return new THREE.CanvasTexture(canvas);
    }

    function createStarlinkGoldFoilBumpTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#808080";
        ctx.fillRect(0, 0, 512, 512);

        for (let i = 0; i < 400; i++) {
            const x1 = Math.random() * 512;
            const y1 = Math.random() * 512;
            const x2 = x1 + (Math.random() - 0.5) * 70;
            const y2 = y1 + (Math.random() - 0.5) * 70;

            ctx.strokeStyle = "rgba(255, 255, 255, 0.65)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();

            ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x1 + 2, y1 + 2);
            ctx.lineTo(x2 + 2, y2 + 2);
            ctx.stroke();
        }
        return new THREE.CanvasTexture(canvas);
    }

    // 4. 4-Point Star Anamorphic Lens Flare (Matching Image Red Strobe)
    function createStarlinkFlareTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, 256, 256);

        const cx = 128;
        const cy = 128;

        // Soft outer red corona glow
        const radialGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 120);
        radialGlow.addColorStop(0, "rgba(255, 30, 68, 0.95)");
        radialGlow.addColorStop(0.2, "rgba(255, 20, 60, 0.6)");
        radialGlow.addColorStop(0.6, "rgba(255, 0, 40, 0.15)");
        radialGlow.addColorStop(1, "rgba(255, 0, 0, 0)");
        ctx.fillStyle = radialGlow;
        ctx.fillRect(0, 0, 256, 256);

        function drawSpike(angle, length, width, color) {
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(angle);
            const grad = ctx.createLinearGradient(-width / 2, 0, width / 2, 0);
            grad.addColorStop(0, "rgba(255, 255, 255, 0)");
            grad.addColorStop(0.5, color);
            grad.addColorStop(1, "rgba(255, 255, 255, 0)");

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.moveTo(-width / 2, 0);
            ctx.lineTo(0, -length);
            ctx.lineTo(width / 2, 0);
            ctx.lineTo(0, length);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        }

        // Primary Horizontal & Vertical Spikes (Long & Brilliant)
        drawSpike(0, 124, 7, "rgba(255, 230, 240, 0.95)");
        drawSpike(Math.PI / 2, 124, 7, "rgba(255, 230, 240, 0.95)");

        // Secondary Diagonal Spikes
        drawSpike(Math.PI / 4, 60, 4, "rgba(255, 80, 100, 0.7)");
        drawSpike(-Math.PI / 4, 60, 4, "rgba(255, 80, 100, 0.7)");

        // Brilliant White-Hot Center Core
        const coreGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 18);
        coreGlow.addColorStop(0, "rgba(255, 255, 255, 1.0)");
        coreGlow.addColorStop(0.5, "rgba(255, 240, 245, 0.9)");
        coreGlow.addColorStop(1, "rgba(255, 50, 80, 0)");
        ctx.fillStyle = coreGlow;
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.fill();

        return new THREE.CanvasTexture(canvas);
    }

    // 5. Hall-Effect Ion Thruster Plasma Plume Gradient
    function createThrusterPlasmaTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 128;
        canvas.height = 256;
        const ctx = canvas.getContext("2d");
        const grad = ctx.createLinearGradient(64, 0, 64, 256);
        grad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
        grad.addColorStop(0.15, "rgba(56, 189, 248, 0.85)");
        grad.addColorStop(0.5, "rgba(2, 132, 199, 0.45)");
        grad.addColorStop(1, "rgba(30, 58, 138, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(40, 0);
        ctx.lineTo(88, 0);
        ctx.lineTo(110, 250);
        ctx.lineTo(18, 250);
        ctx.closePath();
        ctx.fill();
        return new THREE.CanvasTexture(canvas);
    }

    // ============================================================
    // Active Starlink Satellite (Single Large Solar Array Chassis)
    // Modeled 1:1 after Starlink v1.5 / v2 Mini seen in Image 4
    // ============================================================
    function buildActiveStarlinkSatellite() {
        activeSatelliteGroup = new THREE.Group();

        // 1. Procedural PBR Textures
        const solarTex = createStarlinkSolarTexture();
        const solarBump = createStarlinkSolarBumpTexture();
        const chassisTopTex = createStarlinkChassisTopTexture();
        const goldFoilTex = createStarlinkGoldFoilTexture();
        const goldFoilBump = createStarlinkGoldFoilBumpTexture();
        const flareTex = createStarlinkFlareTexture();
        const plasmaTex = createThrusterPlasmaTexture();

        // 2. Main Bus / Satellite Chassis
        const busWidth = 1.35;
        const busHeight = 0.32;
        const busLength = 2.45;

        // PBR Materials for Chassis
        const sideMat = new THREE.MeshStandardMaterial({
            color: 0xdde2ea,
            metalness: 0.85,
            roughness: 0.3,
        });
        const topMat = new THREE.MeshStandardMaterial({
            map: chassisTopTex,
            metalness: 0.8,
            roughness: 0.28,
        });
        const bottomMat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            metalness: 0.9,
            roughness: 0.2,
        });

        // Chassis Box: [right, left, top, bottom, front, back]
        const busMaterials = [sideMat, sideMat, topMat, bottomMat, sideMat, sideMat];
        const busGeo = new THREE.BoxGeometry(busWidth, busHeight, busLength);
        const busMesh = new THREE.Mesh(busGeo, busMaterials);
        activeSatelliteGroup.add(busMesh);

        // 3. Gold MLI Crinkled Thermal Foil Blanket
        const foilMat = new THREE.MeshStandardMaterial({
            color: 0xd4af37,
            map: goldFoilTex,
            bumpMap: goldFoilBump,
            bumpScale: 0.06,
            metalness: 0.92,
            roughness: 0.22,
        });
        const foilUnderbelly = new THREE.Mesh(new THREE.BoxGeometry(1.37, 0.08, 2.47), foilMat);
        foilUnderbelly.position.y = -busHeight / 2 - 0.02;
        activeSatelliteGroup.add(foilUnderbelly);

        // Gold foil corner skirts & wraps
        const foilAftCradle = new THREE.Mesh(new THREE.BoxGeometry(1.36, 0.25, 0.42), foilMat);
        foilAftCradle.position.set(0, -0.06, -busLength / 2 + 0.18);
        activeSatelliteGroup.add(foilAftCradle);

        // 4. Gold Conduit Harness Pipes (Running along lateral edges)
        const conduitMat = new THREE.MeshStandardMaterial({
            color: 0xeab308,
            metalness: 0.95,
            roughness: 0.15,
        });
        [-busWidth / 2 - 0.02, busWidth / 2 + 0.02].forEach((xPos) => {
            const pipeGeo = new THREE.CylinderGeometry(0.025, 0.025, 2.15, 8);
            const pipe = new THREE.Mesh(pipeGeo, conduitMat);
            pipe.rotation.x = Math.PI / 2;
            pipe.position.set(xPos, 0.06, 0);
            activeSatelliteGroup.add(pipe);

            // Bracket clamps
            for (let z = -0.8; z <= 0.8; z += 0.4) {
                const clamp = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.06, 0.04), new THREE.MeshStandardMaterial({ color: 0x334155 }));
                clamp.position.set(xPos, 0.06, z);
                activeSatelliteGroup.add(clamp);
            }
        });

        // 5. Dual Hall-Effect Krypton/Argon Ion Thrusters (Stern)
        const thrusterMountMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9, roughness: 0.3 });
        const thrusterNozzleMat = new THREE.MeshStandardMaterial({
            color: 0x334155,
            metalness: 0.95,
            roughness: 0.15,
        });
        const anodeGlowMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });

        const thrusterSpacing = 0.38;
        activeSatPlumes = [];

        [-thrusterSpacing, thrusterSpacing].forEach((xOffset) => {
            // Bracket base
            const bracket = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.18, 0.15), thrusterMountMat);
            bracket.position.set(xOffset, -0.08, -busLength / 2 - 0.07);
            activeSatelliteGroup.add(bracket);

            // Conical Nozzle Bell
            const nozzleGeo = new THREE.CylinderGeometry(0.12, 0.18, 0.22, 16, 1, true);
            const nozzle = new THREE.Mesh(nozzleGeo, thrusterNozzleMat);
            nozzle.rotation.x = Math.PI / 2;
            nozzle.position.set(xOffset, -0.08, -busLength / 2 - 0.18);
            activeSatelliteGroup.add(nozzle);

            // Inner Anode Ring
            const anode = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.02, 16), anodeGlowMat);
            anode.rotation.x = Math.PI / 2;
            anode.position.set(xOffset, -0.08, -busLength / 2 - 0.14);
            activeSatelliteGroup.add(anode);

            // Ion Thruster Plasma Plume (Exhaust glow)
            const plumeGeo = new THREE.ConeGeometry(0.16, 0.72, 16, 1, true);
            const plumeMat = new THREE.MeshBasicMaterial({
                map: plasmaTex,
                transparent: true,
                opacity: 0.85,
                blending: THREE.AdditiveBlending,
                side: THREE.DoubleSide,
            });
            const plume = new THREE.Mesh(plumeGeo, plumeMat);
            plume.rotation.x = -Math.PI / 2;
            plume.position.set(xOffset, -0.08, -busLength / 2 - 0.54);
            activeSatelliteGroup.add(plume);
            activeSatPlumes.push(plume);
        });

        // 6. Giant Single Vertical Solar Array Wing (Signature Starlink Design!)
        const solarGroup = new THREE.Group();
        solarGroup.position.set(0, busHeight / 2, -0.4);

        // Articulated Solar Array Drive Hinge (SADA)
        const hingeCylinder = new THREE.Mesh(
            new THREE.CylinderGeometry(0.08, 0.08, 0.45, 16),
            new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.2 })
        );
        hingeCylinder.rotation.z = Math.PI / 2;
        hingeCylinder.position.set(0, 0.08, 0);
        solarGroup.add(hingeCylinder);

        // Structural Support Struts
        const strutMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.85 });
        const leftStrut = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.35, 8), strutMat);
        leftStrut.position.set(-0.25, 0.18, 0);
        leftStrut.rotation.z = -0.3;
        solarGroup.add(leftStrut);

        const rightStrut = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.35, 8), strutMat);
        rightStrut.position.set(0.25, 0.18, 0);
        rightStrut.rotation.z = 0.3;
        solarGroup.add(rightStrut);

        // Solar Array Wing Mesh
        const wingWidth = 1.15;
        const wingHeight = 3.6;
        const wingThick = 0.028;

        const solarFrontMat = new THREE.MeshStandardMaterial({
            map: solarTex,
            bumpMap: solarBump,
            bumpScale: 0.03,
            metalness: 0.85,
            roughness: 0.22,
        });

        const solarBackMat = new THREE.MeshStandardMaterial({
            color: 0x0f172a,
            metalness: 0.6,
            roughness: 0.6,
        });

        const frameEdgeMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            metalness: 0.9,
            roughness: 0.3,
        });

        // Wing Box Materials: [right, left, top, bottom, front, back]
        const wingMaterials = [frameEdgeMat, frameEdgeMat, frameEdgeMat, frameEdgeMat, solarFrontMat, solarBackMat];
        const wingMesh = new THREE.Mesh(new THREE.BoxGeometry(wingWidth, wingHeight, wingThick), wingMaterials);
        wingMesh.position.set(0, wingHeight / 2 + 0.2, 0);
        solarGroup.add(wingMesh);

        // Angle the solar array upward and back (~72° tilt) exactly as in the user's reference image!
        solarGroup.rotation.x = -0.35;
        solarGroup.rotation.y = 0.12;
        activeSatelliteGroup.add(solarGroup);
        activeSatSolarWing = solarGroup;

        // 7. Optical Inter-Satellite Laser Communications Downlink Turret
        const laserTurretGroup = new THREE.Group();
        laserTurretGroup.position.set(0.3, -busHeight / 2 - 0.06, 0.6);

        const turretBase = new THREE.Mesh(
            new THREE.CylinderGeometry(0.12, 0.14, 0.08, 16),
            new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9, roughness: 0.2 })
        );
        laserTurretGroup.add(turretBase);

        const turretSphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.11, 16, 16),
            new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.92, roughness: 0.15 })
        );
        turretSphere.position.y = -0.06;
        laserTurretGroup.add(turretSphere);

        // Ruby Laser Aperture Lens
        const lensGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.03, 16);
        const lensMat = new THREE.MeshStandardMaterial({
            color: 0xef4444,
            emissive: 0xef4444,
            emissiveIntensity: 0.9,
            metalness: 0.95,
            roughness: 0.05,
        });
        const lensMesh = new THREE.Mesh(lensGeo, lensMat);
        lensMesh.position.set(0, -0.15, 0.04);
        lensMesh.rotation.x = 0.4;
        laserTurretGroup.add(lensMesh);
        activeSatelliteGroup.add(laserTurretGroup);

        // 8. Forward Star Tracker Sensor Hoods & Nose Detail
        const noseMat = new THREE.MeshStandardMaterial({ color: 0xc8d0dc, metalness: 0.85, roughness: 0.3 });
        const noseBevel = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.2, 0.3), noseMat);
        noseBevel.position.set(0, 0.02, busLength / 2 + 0.12);
        noseBevel.rotation.x = 0.25;
        activeSatelliteGroup.add(noseBevel);

        // Star Tracker Cameras
        [-0.32, 0.32].forEach((x) => {
            const trackerHood = new THREE.Mesh(
                new THREE.CylinderGeometry(0.04, 0.06, 0.1, 12, 1, true),
                new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.95 })
            );
            trackerHood.rotation.x = -Math.PI / 4;
            trackerHood.position.set(x, busHeight / 2 + 0.05, busLength / 2 + 0.05);
            activeSatelliteGroup.add(trackerHood);
        });

        // 9. High-Intensity Flashing Red Strobe Beacon with 4-Point Star Lens Flare!
        // Positioned at top-right forward corner exactly as in user's image
        const strobeX = busWidth / 2 - 0.12;
        const strobeY = busHeight / 2 + 0.06;
        const strobeZ = busLength / 2 - 0.25;

        // Machined Beacon Base
        const beaconBase = new THREE.Mesh(
            new THREE.CylinderGeometry(0.055, 0.07, 0.06, 12),
            new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.95, roughness: 0.1 })
        );
        beaconBase.position.set(strobeX, strobeY, strobeZ);
        activeSatelliteGroup.add(beaconBase);

        // Beacon Bulb
        const bulbGeo = new THREE.SphereGeometry(0.045, 12, 12);
        const bulbMat = new THREE.MeshBasicMaterial({ color: 0xff0033 });
        activeSatStrobe = new THREE.Mesh(bulbGeo, bulbMat);
        activeSatStrobe.position.set(strobeX, strobeY + 0.05, strobeZ);
        activeSatelliteGroup.add(activeSatStrobe);

        // Dynamic Red Point Light illuminating chassis
        activeSatBeaconLight = new THREE.PointLight(0xff0033, 2.8, 5.0);
        activeSatBeaconLight.position.set(strobeX, strobeY + 0.06, strobeZ);
        activeSatelliteGroup.add(activeSatBeaconLight);

        // 4-Point Star Anamorphic Lens Flare Sprite!
        const flareMat = new THREE.SpriteMaterial({
            map: flareTex,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });
        activeSatFlareSprite = new THREE.Sprite(flareMat);
        activeSatFlareSprite.scale.set(1.4, 1.4, 1.4);
        activeSatFlareSprite.position.set(strobeX, strobeY + 0.06, strobeZ);
        activeSatelliteGroup.add(activeSatFlareSprite);

        // Initial Orbit Scale
        activeSatelliteGroup.scale.set(0.95, 0.95, 0.95);
        scene.add(activeSatelliteGroup);
    }

    // ============================================================
    // Laser Beam, Ground Uplink & Neural Computation Systems
    // ============================================================
    function buildLaserTargetingSystem() {
        // --------------------------------------------------------
        // 1. Scene 1: Ground Uplink Beam & Atmospheric Ionization Aura
        // --------------------------------------------------------
        const upGeo = new THREE.CylinderGeometry(0.04, 0.04, 1, 16, 1, true);
        const upMat = new THREE.MeshBasicMaterial({
            color: 0x00f0ff,
            transparent: true,
            opacity: 0,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide,
        });
        uplinkBeamMesh = new THREE.Mesh(upGeo, upMat);
        uplinkBeamMesh.visible = false;
        scene.add(uplinkBeamMesh);

        const upAuraGeo = new THREE.CylinderGeometry(0.14, 0.14, 1, 16, 1, true);
        const upAuraMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide,
        });
        uplinkAuraMesh = new THREE.Mesh(upAuraGeo, upAuraMat);
        uplinkAuraMesh.visible = false;
        scene.add(uplinkAuraMesh);

        // Ground Launch Concentric Rings (Radiating upwards from Earth)
        groundLaunchGroup = new THREE.Group();
        launchRings = [];
        for (let i = 0; i < 4; i++) {
            const ringGeo = new THREE.RingGeometry(0.08, 0.14, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x00f0ff,
                transparent: true,
                opacity: 0,
                side: THREE.DoubleSide,
                blending: THREE.AdditiveBlending,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.visible = false;
            groundLaunchGroup.add(ring);
            launchRings.push({ mesh: ring, progress: -i * 0.25 });
        }
        scene.add(groundLaunchGroup);

        // Traveling Uplink Data Packets (Earth -> Satellite)
        uplinkPackets = [];
        for (let i = 0; i < 8; i++) {
            const pktGeo = new THREE.SphereGeometry(0.085, 8, 8);
            const pktMat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.95,
                blending: THREE.AdditiveBlending,
            });
            const pkt = new THREE.Mesh(pktGeo, pktMat);
            pkt.visible = false;
            pkt.userData = { prog: (i * 0.125) % 1.0 };
            scene.add(pkt);
            uplinkPackets.push(pkt);
        }

        // --------------------------------------------------------
        // 2. Scene 2: Satellite Neural Computing Halo & Processor Light
        // --------------------------------------------------------
        if (activeSatelliteGroup) {
            satelliteNeuralHalo = new THREE.Group();

            const ring1Geo = new THREE.TorusGeometry(0.75, 0.024, 12, 48);
            const ring1Mat = new THREE.MeshBasicMaterial({
                color: 0x38bdf8,
                transparent: true,
                opacity: 0.85,
                blending: THREE.AdditiveBlending,
            });
            const ring1 = new THREE.Mesh(ring1Geo, ring1Mat);
            satelliteNeuralHalo.add(ring1);
            satelliteComputeRings.push(ring1);

            const ring2Geo = new THREE.TorusGeometry(0.52, 0.018, 12, 40);
            const ring2Mat = new THREE.MeshBasicMaterial({
                color: 0xf59e0b,
                transparent: true,
                opacity: 0.9,
                blending: THREE.AdditiveBlending,
            });
            const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
            ring2.rotation.x = Math.PI / 4;
            satelliteNeuralHalo.add(ring2);
            satelliteComputeRings.push(ring2);

            activeSatComputeLight = new THREE.PointLight(0x38bdf8, 0, 14);
            satelliteNeuralHalo.add(activeSatComputeLight);

            satelliteNeuralHalo.visible = false;
            activeSatelliteGroup.add(satelliteNeuralHalo);
        }

        // --------------------------------------------------------
        // 3. Scene 3: Satellite Return Laser Downlink & Ground Impact Rings
        // --------------------------------------------------------
        const beamGeo = new THREE.CylinderGeometry(0.045, 0.065, 1, 16, 1, true);
        const beamMat = new THREE.MeshBasicMaterial({
            color: 0xef4444,
            transparent: true,
            opacity: 0,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide,
        });
        laserBeamMesh = new THREE.Mesh(beamGeo, beamMat);
        laserBeamMesh.visible = false;
        scene.add(laserBeamMesh);

        const downAuraGeo = new THREE.CylinderGeometry(0.18, 0.22, 1, 16, 1, true);
        const downAuraMat = new THREE.MeshBasicMaterial({
            color: 0xef4444,
            transparent: true,
            opacity: 0,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide,
        });
        downlinkAuraMesh = new THREE.Mesh(downAuraGeo, downAuraMat);
        downlinkAuraMesh.visible = false;
        scene.add(downlinkAuraMesh);

        // Traveling Downlink Sparks along the beam
        laserSparks = [];
        for (let i = 0; i < 8; i++) {
            const sparkGeo = new THREE.SphereGeometry(0.08, 8, 8);
            const sparkMat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.95,
                blending: THREE.AdditiveBlending,
            });
            const spark = new THREE.Mesh(sparkGeo, sparkMat);
            spark.visible = false;
            spark.userData = { prog: (i * 0.125) % 1.0 };
            scene.add(spark);
            laserSparks.push(spark);
        }

        // Concentric Shockwave Rings at Target Ground Zero
        groundPingGroup = new THREE.Group();
        pingRings = [];
        for (let i = 0; i < 5; i++) {
            const ringGeo = new THREE.RingGeometry(0.08, 0.14, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0xef4444,
                transparent: true,
                opacity: 0,
                side: THREE.DoubleSide,
                blending: THREE.AdditiveBlending,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.visible = false;
            groundPingGroup.add(ring);
            pingRings.push({ mesh: ring, progress: 0 });
        }
        scene.add(groundPingGroup);
    }

    // ============================================================
    // Persistent Ground Markers (Matching Image 4)
    // 1. Red Target Reticle with Leader Line to HUD Box
    // 2. Amber Location Tag: TKV - 945K / Shane Woods, WA
    // ============================================================
    // ============================================================
    // Persistent Ground Markers (Strictly Project Locations)
    // Pins ONLY California, Texas, New York, Florida
    // ============================================================
    function buildGroundMarkers() {
        groundMarkerGroup = new THREE.Group();
        earthMesh.add(groundMarkerGroup);

        // 1. Primary Texas Target Reticle (Default Image 4 Project Target)
        const primaryLoc = PROJECT_LOCATIONS["texas"];
        const pPos = latLonToVector3(primaryLoc.lat, primaryLoc.lon, GLOBE_RADIUS);

        const reticleGeo = new THREE.RingGeometry(0.12, 0.16, 24);
        const reticleMat = new THREE.MeshBasicMaterial({
            color: 0xef4444,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85,
        });
        const reticleMesh = new THREE.Mesh(reticleGeo, reticleMat);
        reticleMesh.position.copy(pPos);
        reticleMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), pPos.clone().normalize());
        groundMarkerGroup.add(reticleMesh);

        // 2. Secondary Amber Beacon on California (Silicon Valley Tech Hub)
        const caLoc = PROJECT_LOCATIONS["california"];
        const caPos = latLonToVector3(caLoc.lat, caLoc.lon, GLOBE_RADIUS);
        const caDotGeo = new THREE.SphereGeometry(0.065, 8, 8);
        const caDotMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
        const caDot = new THREE.Mesh(caDotGeo, caDotMat);
        caDot.position.copy(caPos);
        groundMarkerGroup.add(caDot);

        // 3. Mark ALL 4 Project Locations with high-visibility glowing beacon pins
        Object.keys(PROJECT_LOCATIONS).forEach((key) => {
            const loc = PROJECT_LOCATIONS[key];
            const pos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS);

            // Light needle
            const needleGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.45, 6);
            const needleMat = new THREE.MeshBasicMaterial({
                color: key === "texas" ? 0xef4444 : 0x38bdf8,
                transparent: true,
                opacity: 0.85,
            });
            const needle = new THREE.Mesh(needleGeo, needleMat);
            needle.position.copy(pos.clone().multiplyScalar(1.035));
            needle.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pos.clone().normalize());
            groundMarkerGroup.add(needle);

            // Glowing base ring
            const baseGeo = new THREE.RingGeometry(0.035, 0.075, 16);
            const baseMat = new THREE.MeshBasicMaterial({
                color: key === "texas" ? 0xef4444 : 0x38bdf8,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.9,
            });
            const baseMesh = new THREE.Mesh(baseGeo, baseMat);
            baseMesh.position.copy(pos.clone().multiplyScalar(1.002));
            baseMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), pos.clone().normalize());
            groundMarkerGroup.add(baseMesh);
        });
    }

    // ============================================================
    // Cinematic Movie Simulation Engine (Earth Uplink -> Satellite AI -> Downlink)
    // ============================================================
    function updateMovieHudScene(sceneNum, seq) {
        const letterbox = document.getElementById("satMovieLetterbox");
        const movieSceneTag = document.getElementById("movieSceneTag");
        const movieSubPhase = document.getElementById("movieSubPhase");
        const movieSubText = document.getElementById("movieSubText");
        const movieTxIdDisplay = document.getElementById("movieTxIdDisplay");
        const movieLocDisplay = document.getElementById("movieLocDisplay");
        const movieDecisionDisplay = document.getElementById("movieDecisionDisplay");
        const movieTelemetryTag = document.getElementById("movieTelemetryTag");

        if (letterbox) letterbox.style.display = "flex";
        if (movieTxIdDisplay) movieTxIdDisplay.textContent = seq.txId;
        if (movieLocDisplay) movieLocDisplay.textContent = seq.loc.city;

        if (sceneNum === 1) {
            if (movieSceneTag) movieSceneTag.textContent = "SCENE 01: GROUND TRANSACTION UPLINK";
            if (movieSubPhase) movieSubPhase.textContent = "PHASE 1: TRANSACTION INGESTION & UPLINK";
            if (movieSubText) movieSubText.textContent = `Transmitting 11 engineered features (Amount: ${seq.amount}, Spending Deviation, Location Deviation, Balance Ratio, Device Match) from ${seq.loc.city} terminal to satellite AI node...`;
            if (movieDecisionDisplay) {
                movieDecisionDisplay.textContent = "TRANSMITTING TELEMETRY...";
                movieDecisionDisplay.className = "text-cyan";
            }
            if (movieTelemetryTag) movieTelemetryTag.textContent = "FASTAPI REST • INLINE LATENCY: 14ms • THRESHOLD: T=0.42";
        } else if (sceneNum === 2) {
            if (movieSceneTag) movieSceneTag.textContent = "SCENE 02: REAL-TIME ML INFERENCE";
            if (movieSubPhase) movieSubPhase.textContent = "PHASE 2: REAL-TIME ML INFERENCE";
            if (movieSubText) movieSubText.textContent = `Executing trained pipeline (Random Forest & Logistic Regression with SMOTE/ROS) on 11 behavioral features at T=0.42 cutoff...`;
            if (movieDecisionDisplay) {
                movieDecisionDisplay.textContent = "SCORING MODEL INLINE...";
                movieDecisionDisplay.className = "text-yellow";
            }
            if (movieTelemetryTag) movieTelemetryTag.textContent = "FASTAPI REST • ML MODEL: RF + LOGISTIC REGRESSION";
        } else if (sceneNum === 3) {
            if (movieSceneTag) movieSceneTag.textContent = "SCENE 03: PREDICTION RETURN & INTERCEPTION";
            if (movieSubPhase) movieSubPhase.textContent = seq.isFraud ? "PHASE 3: SUSPICIOUS ACTIVITY INTERCEPTED" : "PHASE 3: TRANSACTION CLEARED";
            if (movieSubText) {
                movieSubText.textContent = seq.isFraud
                    ? `SUSPICIOUS ACTIVITY FLAG = 1 (Risk: ${seq.score}% ≥ 42.0) — AUTOMATED TRANSACTION FREEZE EXECUTED • RECORDED IN POSTGRESQL AUDIT`
                    : `SUSPICIOUS ACTIVITY FLAG = 0 (Risk: ${seq.score}% < 42.0) — TRANSACTION VERIFIED & CLEARED FOR SETTLEMENT`;
            }
            if (movieDecisionDisplay) {
                movieDecisionDisplay.textContent = seq.isFraud ? "FLAGGED: FRAUD INTERCEPTED" : "CLEARED: LEGITIMATE PAYMENT";
                movieDecisionDisplay.className = seq.isFraud ? "text-danger" : "text-green";
            }
            if (movieTelemetryTag) movieTelemetryTag.textContent = seq.isFraud ? "STATUS: SUSPICIOUS ACTIVITY FLAG = 1" : "STATUS: SUSPICIOUS ACTIVITY FLAG = 0";
        }
    }

    function playCinematicMovie(txData) {
        if (!scene || !earthMesh || !activeSatelliteGroup) return;

        const loc = resolveLocation(txData?.location || txData?.city || "Texas");
        const isFraud = txData?.prediction === "Fraud" || txData?.prediction === "Suspicious" || (txData?.risk_score !== undefined && Number(txData.risk_score) >= 42) || Boolean(txData?.is_fraud);
        const txId = txData?.transaction_id || `T${Math.floor(1000 + Math.random() * 9000)}`;
        const amount = Number(txData?.amount || 8450.0).toLocaleString("en-US", { style: "currency", currency: "USD" });
        const score = txData?.risk_score !== undefined ? Number(txData.risk_score).toFixed(1) : (isFraud ? "89.0" : "4.2");

        const localTargetPos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS);
        const worldTargetPos = localTargetPos.clone().applyEuler(earthMesh.rotation);

        // Position Active Satellite directly above target prograde
        currentOrbitAngle = Math.atan2(worldTargetPos.z, worldTargetPos.x) + 0.38;

        activeMovieSequence = {
            totalDuration: 380,
            frame: 0,
            phase: 1,
            loc: loc,
            isFraud: isFraud,
            txId: txId,
            amount: amount,
            score: score,
            localTargetPos: localTargetPos,
            worldTargetPos: worldTargetPos,
        };

        // Initialize HUD & Scene 1 Audio
        updateMovieHudScene(1, activeMovieSequence);
        playMovieAudio("uplink");

        // Smoothly rotate Earth so target faces viewer
        const targetRotY = -(loc.lon * Math.PI) / 180 + Math.PI * 0.95;
        smoothRotateEarthTo(targetRotY, 1200);
    }

    function triggerSatelliteInterception(txData) {
        playCinematicMovie(txData);
    }

    // Smooth Earth Rotation Helper
    function smoothRotateEarthTo(destY, durationMs) {
        const startY = earthMesh.rotation.y;
        const startTime = performance.now();
        globeRotationSpeed = 0; // Pause auto-spin during targeted strike

        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / durationMs, 1.0);
            const ease = 0.5 - Math.cos(progress * Math.PI) / 2;
            earthMesh.rotation.y = startY + (destY - startY) * ease;
            if (progress < 1.0) {
                requestAnimationFrame(step);
            } else {
                globeRotationSpeed = 0.0016;
            }
        }
        requestAnimationFrame(step);
    }

    // ============================================================
    // Image 4 Floating Target Callout Card
    // "Chance of collision: 89%" / "Est. time: 15:31:11"
    // ============================================================
    function updateImage4HudCard(loc, txId, amount, score, isFraud) {
        let hud = document.getElementById("satImage4HudCard");
        if (!hud) {
            hud = document.createElement("div");
            hud.id = "satImage4HudCard";
            hud.className = "sat-image4-hud-card";
            container.appendChild(hud);
        }

        const sevClass = isFraud ? "danger" : "clean";
        const probText = isFraud ? `SUSPICIOUS RISK: ${score}%` : `LEGITIMATE VERIFIED: ${score}%`;
        const flagText = isFraud ? `FLAG: 1 (FRAUD DETECTED)` : `FLAG: 0 (NORMAL)`;
        const timeStr = new Date().toLocaleTimeString();

        hud.innerHTML = `
            <div class="hud-leader-line"></div>
            <div class="hud-box-inner ${sevClass}">
                <div class="hud-top-line">
                    <span class="hud-title-txt">${probText}</span>
                    <span class="hud-status-dot ${sevClass}"></span>
                </div>
                <div class="hud-bottom-line">
                    <span>${flagText}</span>
                    <span class="hud-sub-loc">${loc.city}</span>
                </div>
                <div class="hud-tx-strip">
                    <span class="font-mono text-muted">${txId}</span>
                    <strong class="font-mono text-white">${amount}</strong>
                </div>
            </div>
        `;

        hud.style.display = "block";
        hud.classList.remove("fade-out");

        clearTimeout(hud._timer);
        hud._timer = setTimeout(() => {
            hud.classList.add("fade-out");
            setTimeout(() => { hud.style.display = "none"; }, 500);
        }, 5000);
    }

    // Position HUD card in 2D screen space above target coordinates
    function updateHudScreenProjection() {
        const hud = document.getElementById("satImage4HudCard");
        if (!hud || hud.style.display === "none") return;

        // Position either on active target or on default primary target
        let targetPos;
        if (activeInterception) {
            targetPos = activeInterception.localTargetPos.clone().applyEuler(earthMesh.rotation);
        } else {
            const loc = PROJECT_LOCATIONS["texas"];
            targetPos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS).applyEuler(earthMesh.rotation);
        }

        targetPos.project(camera);

        // Behind Earth check
        if (targetPos.z > 1.0) {
            hud.style.opacity = "0.2";
            return;
        } else {
            hud.style.opacity = "1";
        }

        const wHalf = (container.clientWidth || 920) / 2;
        const hHalf = (container.clientHeight || 560) / 2;

        const screenX = targetPos.x * wHalf + wHalf;
        const screenY = -(targetPos.y * hHalf) + hHalf;

        hud.style.left = `${Math.min(Math.max(screenX + 18, 20), (container.clientWidth || 920) - 240)}px`;
        hud.style.top = `${Math.min(Math.max(screenY - 50, 20), (container.clientHeight || 560) - 130)}px`;
    }

    // ============================================================
    // Telemetry & UI Decks Sync
    // ============================================================
    function updateHeaderAndDeckTelemetry(loc, txId, isFraud) {
        const satStatusPill = document.getElementById("satStatusPill");
        if (satStatusPill) {
            satStatusPill.textContent = isFraud ? "INTERCEPTING THREAT" : "READY // MONITORING";
            satStatusPill.className = `sat-telemetry-badge ${isFraud ? 'danger' : 'success'}`;
        }

        // Add row to location tracking docket matching Image 4 columns
        const tableBody = document.getElementById("satLocationDocketBody");
        if (tableBody) {
            const row = document.createElement("tr");
            row.className = "towers-row";
            const loadPct = loc.load || (isFraud ? 95 : 45);
            const loadClass = isFraud ? "fill-danger" : (loadPct > 70 ? "fill-warning" : "");
            row.innerHTML = `
                <td><span class="tower-dot ${isFraud ? 'danger' : 'warning'}"></span> <strong class="font-mono">${loc.tag}</strong></td>
                <td>${loc.city}</td>
                <td>
                    <div class="tower-load-bar"><div class="load-fill ${loadClass}" style="width: ${loadPct}%;"></div></div>
                    <small class="font-mono">${loadPct}%</small>
                </td>
                <td class="font-mono">${(loc.devices || 1400).toLocaleString()}</td>
            `;
            tableBody.insertBefore(row, tableBody.firstChild);
            if (tableBody.children.length > 4) tableBody.removeChild(tableBody.lastChild);
        }
    }

    // ============================================================
    // Interactions (Mouse Drag, Scroll Zoom, D-Pad Gimbal)
    // ============================================================
    function initInteractions() {
        if (!container) return;

        container.addEventListener("mousedown", (e) => {
            isUserDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener("mouseup", () => {
            isUserDragging = false;
        });

        container.addEventListener("mousemove", (e) => {
            if (!isUserDragging || !earthMesh) return;
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            earthMesh.rotation.y += deltaX * 0.005;
            earthMesh.rotation.x += deltaY * 0.005;
            earthMesh.rotation.x = Math.max(-Math.PI / 4, Math.min(Math.PI / 4, earthMesh.rotation.x));
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        container.addEventListener("wheel", (e) => {
            e.preventDefault();
            camera.position.z += e.deltaY * 0.015;
            camera.position.z = Math.max(14, Math.min(32, camera.position.z));
        }, { passive: false });

        // D-Pad Gimbal Controls (Bottom-Left Deck)
        const dpadUp = document.getElementById("btnDpadUp");
        const dpadDown = document.getElementById("btnDpadDown");
        const dpadLeft = document.getElementById("btnDpadLeft");
        const dpadRight = document.getElementById("btnDpadRight");
        const dpadCenter = document.getElementById("btnDpadCenter");

        if (dpadUp) dpadUp.addEventListener("click", () => { if (earthMesh) earthMesh.rotation.x += 0.12; });
        if (dpadDown) dpadDown.addEventListener("click", () => { if (earthMesh) earthMesh.rotation.x -= 0.12; });
        if (dpadLeft) dpadLeft.addEventListener("click", () => { if (earthMesh) earthMesh.rotation.y -= 0.15; });
        if (dpadRight) dpadRight.addEventListener("click", () => { if (earthMesh) earthMesh.rotation.y += 0.15; });
        if (dpadCenter) dpadCenter.addEventListener("click", () => {
            if (earthMesh) { earthMesh.rotation.x = 0; earthMesh.rotation.y = 1.35; }
            camera.position.set(0, 4.5, 23);
        });

        // Cinematic Movie Play Button & Test Fire Button
        const sampleFrauds = [
            { transaction_id: "T9896", location: "Texas", amount: 1508.20, risk_score: 89.0, prediction: "Fraud" },
            { transaction_id: "T3227", location: "California", amount: 91.62, risk_score: 96.4, prediction: "Fraud" },
            { transaction_id: "T6414", location: "Florida", amount: 9593.98, risk_score: 98.7, prediction: "Fraud" },
            { transaction_id: "T6711", location: "New York", amount: 3921.94, risk_score: 91.2, prediction: "Fraud" },
            { transaction_id: "T6369", location: "California", amount: 1712.69, risk_score: 94.1, prediction: "Fraud" },
            { transaction_id: "T7768", location: "New York", amount: 4852.90, risk_score: 93.5, prediction: "Fraud" },
        ];

        const btnPlayMovie = document.getElementById("btnPlayMovieInterception");
        if (btnPlayMovie) {
            btnPlayMovie.addEventListener("click", () => {
                const picked = sampleFrauds[Math.floor(Math.random() * sampleFrauds.length)];
                playCinematicMovie(picked);
            });
        }

        const btnLaunchMission = document.getElementById("btnLaunchMissionPrediction");
        if (btnLaunchMission) {
            btnLaunchMission.addEventListener("click", () => {
                const locSelect = document.getElementById("missionSelectLoc");
                const scenSelect = document.getElementById("missionSelectScenario");
                const loc = locSelect ? locSelect.value : "texas";
                const scen = scenSelect ? scenSelect.value : "attack_drain";

                let txId, amount, risk_score, prediction, isFraud;
                const rnd = Math.floor(1000 + Math.random() * 9000);

                if (scen === "attack_drain") {
                    txId = `T${rnd}`;
                    amount = 1508.20;
                    risk_score = 100.0;
                    prediction = "Fraud";
                    isFraud = true;
                } else if (scen === "attack_rapid") {
                    txId = `T${rnd}`;
                    amount = 9500.00;
                    risk_score = 98.7;
                    prediction = "Fraud";
                    isFraud = true;
                } else if (scen === "clean_payroll") {
                    txId = `T${rnd}`;
                    amount = 4200.00;
                    risk_score = 8.5;
                    prediction = "Legitimate";
                    isFraud = false;
                } else {
                    txId = `T${rnd}`;
                    amount = 45.00;
                    risk_score = 3.2;
                    prediction = "Legitimate";
                    isFraud = false;
                }

                // Trigger cinematic movie sequence: Earth Uplink -> Orbital AI -> Satellite Downlink
                playCinematicMovie({
                    transaction_id: txId,
                    location: loc,
                    amount: amount,
                    risk_score: risk_score,
                    prediction: prediction,
                    is_fraud: isFraud
                });

                // Update Mission Verdict Display
                const verdictCard = document.getElementById("missionVerdictCard");
                const vBadge = document.getElementById("mVerdictBadge");
                const vAction = document.getElementById("mVerdictAction");
                const vScore = document.getElementById("mVerdictScore");
                const vFlag = document.getElementById("mVerdictFlag");

                if (verdictCard) {
                    verdictCard.style.display = "flex";
                    if (isFraud) {
                        verdictCard.className = "mission-verdict-card";
                        if (vBadge) {
                            vBadge.className = "m-verdict-badge danger";
                            vBadge.textContent = "SUSPICIOUS (FRAUD INTERCEPTED)";
                        }
                        if (vFlag) {
                            vFlag.className = "font-mono text-danger";
                            vFlag.textContent = "1 (SUSPICIOUS / FRAUD DETECTED)";
                        }
                        if (vAction) {
                            vAction.className = "text-danger";
                            vAction.textContent = "🚨 TRANSACTION FROZEN & POSTGRESQL ALERT RECORDED";
                        }
                        if (vScore) {
                            vScore.className = "font-mono text-danger";
                            vScore.textContent = `${risk_score.toFixed(1)} / 100`;
                        }

                        // Increment active cases counter if present
                        const totalAlertEl = document.getElementById("statTotalAlerts");
                        if (totalAlertEl) {
                            const cur = parseInt(totalAlertEl.textContent) || 0;
                            totalAlertEl.textContent = cur + 1;
                        }
                    } else {
                        verdictCard.className = "mission-verdict-card clean";
                        if (vBadge) {
                            vBadge.className = "m-verdict-badge clean";
                            vBadge.textContent = "CLEARED (LEGITIMATE)";
                        }
                        if (vFlag) {
                            vFlag.className = "font-mono text-green";
                            vFlag.textContent = "0 (LEGITIMATE / NORMAL)";
                        }
                        if (vAction) {
                            vAction.className = "text-green";
                            vAction.textContent = "✔ INSTANT SETTLEMENT RELEASED";
                        }
                        if (vScore) {
                            vScore.className = "font-mono text-green";
                            vScore.textContent = `${risk_score.toFixed(1)} / 100`;
                        }
                    }
                }
            });
        }

        // Camera Modes
        const btnCamOrbital = document.getElementById("btnCamOrbital");
        const btnCamChase = document.getElementById("btnCamChase");
        const btnCamTarget = document.getElementById("btnCamTarget");

        if (btnCamOrbital) {
            btnCamOrbital.addEventListener("click", () => {
                cameraMode = "orbital";
                setCameraModeActive(btnCamOrbital);
            });
        }
        if (btnCamChase) {
            btnCamChase.addEventListener("click", () => {
                cameraMode = "chase";
                setCameraModeActive(btnCamChase);
            });
        }
        if (btnCamTarget) {
            btnCamTarget.addEventListener("click", () => {
                cameraMode = "target";
                setCameraModeActive(btnCamTarget);
            });
        }

        window.addEventListener("resize", onResizeHandler);
    }

    function setCameraModeActive(btn) {
        document.querySelectorAll(".btn-sat-cam").forEach((b) => b.classList.remove("active"));
        if (btn) btn.classList.add("active");

        if (container) {
            if (cameraMode === "chase") {
                container.classList.add("sat-cam-active");
            } else {
                container.classList.remove("sat-cam-active");
            }
        }

        const tacticalHud = document.getElementById("satCamTacticalHud");
        if (tacticalHud) {
            tacticalHud.style.display = cameraMode === "chase" ? "flex" : "none";
        }
    }

    function onResizeHandler() {
        if (!container || !renderer || !camera) return;
        const w = container.clientWidth || 920;
        const h = container.clientHeight || 560;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }

    // ============================================================
    // 60 FPS Render & Animation Loop
    // ============================================================
    let strobeTick = 0;

    function animate() {
        requestAnimationFrame(animate);

        // 1. Continuous Earth Spin
        if (earthMesh && !isUserDragging) {
            earthMesh.rotation.y += globeRotationSpeed;
        }

        // 2. Animate All Constellation Satellites along Orbital Planes
        const satWorldPositions = [];
        constellationSatellites.forEach((sat) => {
            sat.angle += sat.speed;
            const lx = Math.cos(sat.angle) * sat.plane.radius;
            const lz = Math.sin(sat.angle) * sat.plane.radius;
            const v = new THREE.Vector3(lx, 0, lz);
            v.applyEuler(new THREE.Euler(sat.plane.inclX, sat.plane.rotY, sat.plane.inclZ));
            sat.mesh.position.copy(v);
            satWorldPositions.push(v);
        });

        // 3. Update Inter-Satellite Laser Link Lines (ISL Mesh)
        if (interSatLinkLines) {
            const posAttr = interSatLinkLines.geometry.attributes.position;
            let linkCount = 0;
            const maxLinks = 40;

            for (let i = 0; i < satWorldPositions.length && linkCount < maxLinks; i += 2) {
                for (let j = i + 1; j < satWorldPositions.length && linkCount < maxLinks; j++) {
                    const dist = satWorldPositions[i].distanceTo(satWorldPositions[j]);
                    if (dist > 2.0 && dist < 5.2) {
                        const pA = satWorldPositions[i];
                        const pB = satWorldPositions[j];
                        posAttr.setXYZ(linkCount * 2, pA.x, pA.y, pA.z);
                        posAttr.setXYZ(linkCount * 2 + 1, pB.x, pB.y, pB.z);
                        linkCount++;
                    }
                }
            }
            posAttr.needsUpdate = true;
            interSatLinkLines.geometry.setDrawRange(0, linkCount * 2);
        }

        // 4. Animate Active Starlink Satellite Orbit with Authentic Orthonormal Orientation
        if (activeSatelliteGroup) {
            currentOrbitAngle += 0.0032;
            const satX = Math.cos(currentOrbitAngle) * SATELLITE_ORBIT_RADIUS;
            const satZ = Math.sin(currentOrbitAngle) * SATELLITE_ORBIT_RADIUS;
            const satY = Math.sin(currentOrbitAngle) * 3.8;
            activeSatelliteGroup.position.set(satX, satY, satZ);

            // Compute realistic orbital velocity vector (prograde direction)
            const nextAngle = currentOrbitAngle + 0.01;
            const nextX = Math.cos(nextAngle) * SATELLITE_ORBIT_RADIUS;
            const nextZ = Math.sin(nextAngle) * SATELLITE_ORBIT_RADIUS;
            const nextY = Math.sin(nextAngle) * 3.8;
            const forwardVec = new THREE.Vector3(nextX - satX, nextY - satY, nextZ - satZ).normalize();

            // Nadir vector (down to Earth center)
            const nadirVec = new THREE.Vector3(-satX, -satY, -satZ).normalize();
            // Zenith vector (upward into space)
            const zenithVec = nadirVec.clone().negate();
            // Right lateral axis
            const rightVec = new THREE.Vector3().crossVectors(forwardVec, zenithVec).normalize();
            // True orthogonal up vector
            const trueUp = new THREE.Vector3().crossVectors(rightVec, forwardVec).normalize();

            // Set satellite orientation: [right: +X, up: +Y, forward: +Z]
            const rotMatrix = new THREE.Matrix4().makeBasis(rightVec, trueUp, forwardVec);
            activeSatelliteGroup.quaternion.setFromRotationMatrix(rotMatrix);

            // Dynamic Navigation Strobe, Point Light, & 4-Point Star Lens Flare!
            strobeTick += 0.06;
            const isStrobeOn = Math.sin(strobeTick * 6) > 0.35;
            if (activeSatStrobe) {
                activeSatStrobe.material.color.setHex(isStrobeOn ? 0xff0033 : 0x330011);
            }
            if (activeSatBeaconLight) {
                activeSatBeaconLight.intensity = isStrobeOn ? 3.0 : 0.15;
            }
            if (activeSatFlareSprite) {
                activeSatFlareSprite.visible = isStrobeOn;
                if (isStrobeOn) {
                    const flareScale = 1.35 + Math.sin(strobeTick * 12) * 0.25;
                    activeSatFlareSprite.scale.set(flareScale, flareScale, flareScale);
                }
            }

            // Hall-Effect Ion Thruster Plasma Plume Shimmer
            if (activeSatPlumes && activeSatPlumes.length > 0) {
                const plumeScale = 0.85 + Math.sin(strobeTick * 10) * 0.15;
                activeSatPlumes.forEach((p) => {
                    p.scale.set(1, plumeScale, 1);
                    p.material.opacity = 0.75 + Math.sin(strobeTick * 8) * 0.2;
                });
            }
        }

        // 5. Cinematic 3-Phase Movie Sequence Engine (Earth Uplink -> Orbital AI -> Return Downlink)
        if (activeMovieSequence) {
            const seq = activeMovieSequence;
            seq.frame++;

            const currentGroundTarget = seq.localTargetPos
                ? seq.localTargetPos.clone().applyEuler(earthMesh.rotation)
                : seq.worldTargetPos;

            const satPos = activeSatelliteGroup.position.clone();
            const beamDist = satPos.distanceTo(currentGroundTarget);
            const beamMidpoint = satPos.clone().add(currentGroundTarget).multiplyScalar(0.5);
            const beamNormal = currentGroundTarget.clone().sub(satPos).normalize();

            // -------------------------------------------------------------
            // SCENE 1: Ground-to-Satellite Uplink (Frames 1 to 110, ~1.8s)
            // -------------------------------------------------------------
            if (seq.frame <= 110) {
                seq.phase = 1;

                // Ground Launch Concentric Rings radiating upwards from Earth
                groundLaunchGroup.position.copy(currentGroundTarget);
                groundLaunchGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), currentGroundTarget.clone().normalize());
                launchRings.forEach((lr) => {
                    lr.mesh.visible = true;
                    lr.progress += 0.035;
                    const rProg = (lr.progress > 0) ? (lr.progress % 1.0) : 0;
                    lr.mesh.scale.set(1 + rProg * 4.2, 1 + rProg * 4.2, 1);
                    lr.mesh.material.opacity = Math.max(0, 0.95 - rProg * 0.95);
                });

                // Uplink beam grows from Earth upward toward satellite
                const growthProg = Math.min(seq.frame / 75, 1.0);
                const currentUplinkEnd = currentGroundTarget.clone().lerp(satPos, growthProg);
                const upDist = currentGroundTarget.distanceTo(currentUplinkEnd);
                const upMid = currentGroundTarget.clone().add(currentUplinkEnd).multiplyScalar(0.5);
                const upDir = currentUplinkEnd.clone().sub(currentGroundTarget).normalize();

                uplinkBeamMesh.visible = true;
                uplinkAuraMesh.visible = true;
                uplinkBeamMesh.scale.set(1, upDist, 1);
                uplinkAuraMesh.scale.set(1, upDist, 1);
                uplinkBeamMesh.position.copy(upMid);
                uplinkAuraMesh.position.copy(upMid);
                const upQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), upDir);
                uplinkBeamMesh.quaternion.copy(upQuat);
                uplinkAuraMesh.quaternion.copy(upQuat);
                uplinkBeamMesh.material.opacity = 0.95;
                uplinkAuraMesh.material.opacity = 0.35 + Math.sin(seq.frame * 0.2) * 0.15;

                // Uplink packets surge upward from Earth to Space
                uplinkPackets.forEach((pkt) => {
                    pkt.visible = true;
                    pkt.userData.prog = (pkt.userData.prog + 0.045) % 1.0;
                    const pktPos = currentGroundTarget.clone().lerp(satPos, pkt.userData.prog * growthProg);
                    pkt.position.copy(pktPos);
                });

                // Camera Scene 1: Dramatic Low Orbital Launch Perspective (Track upward from Earth to Space)
                const scene1CamTarget = currentGroundTarget.clone().multiplyScalar(1.68).add(new THREE.Vector3(2.6, 2.2, 2.8));
                camera.position.lerp(scene1CamTarget, 0.065);
                const scene1Look = currentGroundTarget.clone().add(satPos.clone().sub(currentGroundTarget).multiplyScalar(0.42));
                camera.lookAt(scene1Look);

                // Ensure downlink elements are hidden during Scene 1
                laserBeamMesh.visible = false;
                downlinkAuraMesh.visible = false;
                laserSparks.forEach((s) => (s.visible = false));
                pingRings.forEach((p) => (p.mesh.visible = false));
                if (activeSatComputeLight) activeSatComputeLight.intensity = 0;
                if (satelliteNeuralHalo) satelliteNeuralHalo.visible = false;

            // -------------------------------------------------------------
            // SCENE 2: Orbital Satellite ML Inference (Frames 111 to 220, ~1.8s)
            // -------------------------------------------------------------
            } else if (seq.frame <= 220) {
                if (seq.phase === 1) {
                    seq.phase = 2;
                    playMovieAudio("inference");
                    updateMovieHudScene(2, seq);
                }

                // Dissolve uplink elements
                uplinkBeamMesh.visible = false;
                uplinkAuraMesh.visible = false;
                uplinkPackets.forEach((p) => (p.visible = false));
                launchRings.forEach((lr) => (lr.mesh.visible = false));

                // Spin satellite neural computation rings
                if (satelliteNeuralHalo) {
                    satelliteNeuralHalo.visible = true;
                    satelliteComputeRings.forEach((ring, idx) => {
                        ring.rotation.x += (idx === 0 ? 0.08 : -0.11);
                        ring.rotation.y += (idx === 0 ? 0.06 : 0.09);
                        ring.rotation.z += 0.04;
                    });
                }

                // Pulse computing processor light
                if (activeSatComputeLight) {
                    activeSatComputeLight.intensity = 2.8 + Math.sin(seq.frame * 0.45) * 1.8;
                    activeSatComputeLight.color.setHex(0x38bdf8);
                }

                // Camera Scene 2: Cinematic Satellite Chase View (Look over satellite wing with Earth below)
                const satRot = activeSatelliteGroup.quaternion;
                const localChaseOffset = new THREE.Vector3(3.2, 1.6, 4.4);
                const worldChaseCam = satPos.clone().add(localChaseOffset.applyQuaternion(satRot));
                camera.position.lerp(worldChaseCam, 0.075);
                const lookSatOffset = new THREE.Vector3(-0.35, 0.2, 0).applyQuaternion(satRot);
                camera.lookAt(satPos.clone().add(lookSatOffset));

            // -------------------------------------------------------------
            // SCENE 3: Downlink Return & Ground Interception (Frames 221 to 350, ~2.2s)
            // -------------------------------------------------------------
            } else if (seq.frame <= 350) {
                if (seq.phase === 2) {
                    seq.phase = 3;
                    playMovieAudio("downlink", seq.isFraud);
                    updateMovieHudScene(3, seq);

                    // Trigger Image 4 floating callout card & location docket
                    updateImage4HudCard(seq.loc, seq.txId, seq.amount, seq.score, seq.isFraud);
                    updateHeaderAndDeckTelemetry(seq.loc, seq.txId, seq.isFraud);
                }

                // Satellite Neural Halo flashes in result color
                const resultColor = seq.isFraud ? 0xef4444 : 0x10b981;
                if (activeSatComputeLight) {
                    activeSatComputeLight.color.setHex(resultColor);
                    activeSatComputeLight.intensity = 3.5;
                }

                // Fire Downlink Return Beam from Satellite to Earth
                laserBeamMesh.visible = true;
                downlinkAuraMesh.visible = true;
                laserBeamMesh.material.color.setHex(resultColor);
                downlinkAuraMesh.material.color.setHex(resultColor);

                laserBeamMesh.scale.set(1, beamDist, 1);
                downlinkAuraMesh.scale.set(1, beamDist, 1);
                laserBeamMesh.position.copy(beamMidpoint);
                downlinkAuraMesh.position.copy(beamMidpoint);
                const downQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), beamNormal);
                laserBeamMesh.quaternion.copy(downQuat);
                downlinkAuraMesh.quaternion.copy(downQuat);
                laserBeamMesh.material.opacity = 1.0;
                downlinkAuraMesh.material.opacity = 0.45;

                // Downward sparks racing toward Earth
                laserSparks.forEach((spark) => {
                    spark.visible = true;
                    spark.material.color.setHex(0xffffff);
                    spark.userData.prog = (spark.userData.prog + 0.05) % 1.0;
                    spark.position.copy(satPos.clone().lerp(currentGroundTarget, spark.userData.prog));
                });

                // Concentric Shockwave Rings on Target Ground Zero
                groundPingGroup.position.copy(currentGroundTarget);
                groundPingGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), currentGroundTarget.clone().normalize());
                pingRings.forEach((p) => {
                    p.mesh.visible = true;
                    p.mesh.material.color.setHex(resultColor);
                    p.progress += 0.03;
                    if (p.progress > 0) {
                        const ringProg = p.progress % 1.0;
                        p.mesh.scale.set(1 + ringProg * 4.8, 1 + ringProg * 4.8, 1);
                        p.mesh.material.opacity = Math.max(0, 1 - ringProg);
                    }
                });

                // Camera Scene 3: Wide Panoramic Orbital Lock (Frames both satellite in high orbit and glowing Earth target)
                const scene3CamTarget = currentGroundTarget.clone().multiplyScalar(2.15).add(new THREE.Vector3(-1.8, 4.2, 4.0));
                camera.position.lerp(scene3CamTarget, 0.045);
                camera.lookAt(currentGroundTarget.clone().add(satPos).multiplyScalar(0.5));

            // -------------------------------------------------------------
            // SCENE 4: Fade-out & Cinematic Wrap-up (Frames 351 to 380)
            // -------------------------------------------------------------
            } else if (seq.frame <= 380) {
                const fadeProg = (seq.frame - 350) / 30;
                laserBeamMesh.material.opacity = Math.max(0, 1 - fadeProg);
                downlinkAuraMesh.material.opacity = Math.max(0, 0.45 * (1 - fadeProg));
                if (activeSatComputeLight) activeSatComputeLight.intensity = Math.max(0, 3.5 * (1 - fadeProg));
                if (satelliteNeuralHalo) satelliteNeuralHalo.visible = false;
                laserSparks.forEach((s) => (s.visible = false));
                pingRings.forEach((p) => (p.mesh.visible = false));

                // Return camera smoothly toward user default
                camera.position.lerp(new THREE.Vector3(0, 4.5, 23), 0.03);
                camera.lookAt(0, 0, 0);

            } else {
                // Complete Sequence
                laserBeamMesh.visible = false;
                downlinkAuraMesh.visible = false;
                activeMovieSequence = null;

                // Auto-hide letterbox after 4 seconds
                const letterbox = document.getElementById("satMovieLetterbox");
                if (letterbox) {
                    setTimeout(() => {
                        if (!activeMovieSequence) letterbox.style.display = "none";
                    }, 4000);
                }
            }

        // Fallback for standard interception if activeMovieSequence is not set
        } else if (activeInterception) {
            activeInterception.frame++;

            const currentGroundTarget = activeInterception.localTargetPos
                ? activeInterception.localTargetPos.clone().applyEuler(earthMesh.rotation)
                : activeInterception.worldTargetPos;

            const satPos = activeSatelliteGroup.position.clone();
            const beamDist = satPos.distanceTo(currentGroundTarget);
            laserBeamMesh.scale.set(1, beamDist, 1);
            laserBeamMesh.position.copy(satPos.clone().add(currentGroundTarget).multiplyScalar(0.5));
            laserBeamMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), currentGroundTarget.clone().sub(satPos).normalize());

            laserSparks.forEach((spark) => {
                spark.userData.prog = (spark.userData.prog + 0.04) % 1.0;
                spark.position.copy(satPos.clone().lerp(currentGroundTarget, spark.userData.prog));
            });

            groundPingGroup.position.copy(currentGroundTarget);
            pingRings.forEach((p) => {
                p.progress += 0.025;
                if (p.progress > 0) {
                    const ringProg = p.progress % 1.0;
                    p.mesh.scale.set(1 + ringProg * 4.5, 1 + ringProg * 4.5, 1);
                    p.mesh.material.opacity = Math.max(0, 1 - ringProg);
                }
            });

            if (activeInterception.frame > activeInterception.duration) {
                laserBeamMesh.visible = false;
                laserSparks.forEach((s) => (s.visible = false));
                pingRings.forEach((p) => (p.mesh.visible = false));
                activeInterception = null;
            }
        }

        // 6. Camera Modes (Only active when movie sequence is not controlling the camera)
        if (!activeMovieSequence) {
            if (cameraMode === "chase" && activeSatelliteGroup) {
                const satPos = activeSatelliteGroup.position.clone();
                const satRot = activeSatelliteGroup.quaternion;
                const localCamOffset = new THREE.Vector3(3.2, 1.6, 4.2);
                const worldCamOffset = localCamOffset.clone().applyQuaternion(satRot);
                const targetCamPos = satPos.clone().add(worldCamOffset);

                camera.position.lerp(targetCamPos, 0.08);
                const localLookOffset = new THREE.Vector3(-0.35, 0.2, 0);
                const worldLookTarget = satPos.clone().add(localLookOffset.applyQuaternion(satRot));
                camera.lookAt(worldLookTarget);
            } else if (cameraMode === "target" && (activeInterception || activeMovieSequence)) {
                const targetRef = activeMovieSequence ? activeMovieSequence.worldTargetPos : activeInterception.worldTargetPos;
                const desiredPos = targetRef.clone().multiplyScalar(2.1);
                camera.position.lerp(desiredPos, 0.04);
                camera.lookAt(targetRef);
            } else {
                camera.lookAt(0, 0, 0);
            }
        }

        // 7. Update Projected 2D HUD Box Location
        updateHudScreenProjection();

        // 8. Render Scene
        renderer.render(scene, camera);
    }

    // ============================================================
    // Global Public API
    // ============================================================
    window.SatelliteDefense = {
        init: initSatelliteTheater,
        triggerInterception: triggerSatelliteInterception,
        playCinematicMovie: playCinematicMovie,
        resolveLocation: resolveLocation,
        onResize: onResizeHandler,
        locations: PROJECT_LOCATIONS,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initSatelliteTheater);
    } else {
        setTimeout(initSatelliteTheater, 100);
    }
})();
