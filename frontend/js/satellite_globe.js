// ============================================================
// Satellite Intelligence & Global Fraud Defense System
// 1:1 Fuselab Creative Starlink Intelligence Interface Engine
// 3D Orbital Earth, Multi-Plane Constellation Mesh & Laser Strike
// ============================================================

(function () {
    "use strict";

    // Known Coordinates for Project Financial Centers & Image 4 Locations
    const FINANCIAL_LOCATIONS = {
        "california": { lat: 36.7783, lon: -119.4179, city: "San Francisco, CA", region: "North America", flag: "🇺🇸", ipPrefix: "240.44", tag: "H18-406" },
        "texas": { lat: 31.9686, lon: -99.9018, city: "Dallas / Austin, TX", region: "North America", flag: "🇺🇸", ipPrefix: "28.75", tag: "TX-770" },
        "new york": { lat: 40.7128, lon: -74.0060, city: "New York, NY", region: "North America", flag: "🇺🇸", ipPrefix: "198.51", tag: "NY-501" },
        "florida": { lat: 27.6648, lon: -81.5158, city: "Miami / Orlando, FL", region: "North America", flag: "🇺🇸", ipPrefix: "172.56", tag: "FL-330" },
        "illinois": { lat: 40.6331, lon: -89.3985, city: "Chicago, IL", region: "North America", flag: "🇺🇸", ipPrefix: "192.0", tag: "CH-112" },
        "washington": { lat: 47.7511, lon: -120.7401, city: "Shane Woods, WA", region: "North America", flag: "🇺🇸", ipPrefix: "204.79", tag: "TKV-945K" },
        "portland": { lat: 45.5152, lon: -122.6784, city: "Portland, OR", region: "North America", flag: "🇺🇸", ipPrefix: "198.22", tag: "T48-001" },
        "seattle": { lat: 47.6062, lon: -122.3321, city: "Seattle, WA", region: "North America", flag: "🇺🇸", ipPrefix: "204.88", tag: "R2-05" },
        "los angeles": { lat: 34.0522, lon: -118.2437, city: "Los Angeles, CA", region: "North America", flag: "🇺🇸", ipPrefix: "173.24", tag: "LA-389" },
        "st louis": { lat: 38.6270, lon: -90.1994, city: "St. Louis, MO", region: "North America", flag: "🇺🇸", ipPrefix: "199.30", tag: "MO-389" },
        "london": { lat: 51.5074, lon: -0.1278, city: "London, UK", region: "Europe", flag: "🇬🇧", ipPrefix: "185.86", tag: "UK-LON" },
        "zurich": { lat: 47.3769, lon: 8.5417, city: "Zurich, Switzerland", region: "Europe", flag: "🇨🇭", ipPrefix: "193.134", tag: "CH-ZUR" },
        "tokyo": { lat: 35.6762, lon: 139.6503, city: "Tokyo, Japan", region: "Asia-Pacific", flag: "🇯🇵", ipPrefix: "133.242", tag: "JP-TYO" },
    };

    // Core Constants
    const GLOBE_RADIUS = 6.2;
    const SATELLITE_ORBIT_RADIUS = 9.8;

    // Three.js Core Objects
    let scene, camera, renderer, container;
    let earthMesh, atmosphereMesh;
    let constellationGroup, constellationSatellites = [], orbitalRings = [];
    let interSatLinkLines;
    let activeSatelliteGroup, activeSatSolarWing, activeSatStrobe;
    let laserBeamMesh, laserSparks = [];
    let groundPingGroup, pingRings = [];
    let groundMarkerGroup;
    let activeInterception = null;

    // Interaction & Animation State
    let isUserDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let globeRotationSpeed = 0.0016;
    let cameraMode = "orbital"; // "orbital", "chase", "target"
    let targetCameraPos = null;
    let targetCameraLook = null;
    let currentOrbitAngle = 0.85;

    // Laser Web Audio Synthesizer
    let audioCtx = null;
    function playLaserAudio(isFraud) {
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === "suspended") audioCtx.resume();
            const now = audioCtx.currentTime;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);

            if (isFraud) {
                // High-priority satellite laser interception chirp
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(1600, now);
                osc.frequency.exponentialRampToValueAtTime(140, now + 0.35);
                gain.gain.setValueAtTime(0.18, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
                osc.start(now);
                osc.stop(now + 0.35);
            } else {
                // Low-risk downlink lock
                osc.type = "sine";
                osc.frequency.setValueAtTime(680, now);
                osc.frequency.exponentialRampToValueAtTime(1250, now + 0.2);
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
                osc.start(now);
                osc.stop(now + 0.2);
            }
        } catch (e) {
            // Audio policy silently handled
        }
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
        if (!locStr) return FINANCIAL_LOCATIONS["california"];
        const lower = String(locStr).toLowerCase().trim();
        for (const key in FINANCIAL_LOCATIONS) {
            if (lower.includes(key)) return FINANCIAL_LOCATIONS[key];
        }
        return FINANCIAL_LOCATIONS["texas"];
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
    // Active Starlink Satellite (Single Large Solar Array Chassis)
    // Modeled 1:1 after Starlink v1.5 / v2 Mini seen in Image 4
    // ============================================================
    function buildActiveStarlinkSatellite() {
        activeSatelliteGroup = new THREE.Group();

        // 1. Flat Rectangular Satellite Chassis (Stacked bus)
        const busGeo = new THREE.BoxGeometry(0.85, 0.22, 1.4);
        const busMat = new THREE.MeshStandardMaterial({
            color: 0xd1d5db,
            metalness: 0.9,
            roughness: 0.25,
        });
        const busMesh = new THREE.Mesh(busGeo, busMat);
        activeSatelliteGroup.add(busMesh);

        // Gold Thermal Foil Underbelly Shield
        const foilGeo = new THREE.BoxGeometry(0.86, 0.05, 1.41);
        const foilMat = new THREE.MeshStandardMaterial({
            color: 0xb4821a,
            emissive: 0x785309,
            emissiveIntensity: 0.35,
            metalness: 0.95,
            roughness: 0.15,
        });
        const foilMesh = new THREE.Mesh(foilGeo, foilMat);
        foilMesh.position.y = -0.11;
        activeSatelliteGroup.add(foilMesh);

        // 2. Single Tall Vertical Solar Wing (Starlink signature!)
        const solarGeo = new THREE.BoxGeometry(0.78, 1.8, 0.03);
        const solarMat = new THREE.MeshStandardMaterial({
            color: 0x1d4ed8,
            emissive: 0x1e40af,
            emissiveIntensity: 0.45,
            metalness: 0.8,
            roughness: 0.2,
        });
        activeSatSolarWing = new THREE.Mesh(solarGeo, solarMat);
        // Erected vertically like real Starlink in Image 4
        activeSatSolarWing.position.set(0, 1.02, -0.65);
        activeSatelliteGroup.add(activeSatSolarWing);

        // Solar Array Boom Hinge
        const boomGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.25, 8);
        const boomMat = new THREE.MeshStandardMaterial({ color: 0x475569 });
        const boom = new THREE.Mesh(boomGeo, boomMat);
        boom.position.set(0, 0.13, -0.65);
        activeSatelliteGroup.add(boom);

        // 3. Optical Laser Communications Terminal (Downlink Emitter)
        const emitterGeo = new THREE.CylinderGeometry(0.12, 0.18, 0.18, 12);
        const emitterMat = new THREE.MeshStandardMaterial({
            color: 0xef4444,
            emissive: 0xef4444,
            emissiveIntensity: 0.8,
        });
        const emitter = new THREE.Mesh(emitterGeo, emitterMat);
        emitter.position.set(0, -0.15, 0.3);
        activeSatelliteGroup.add(emitter);

        // 4. Strobe Navigation Beacon
        const strobeGeo = new THREE.SphereGeometry(0.06, 8, 8);
        const strobeMat = new THREE.MeshBasicMaterial({ color: 0xff0033 });
        activeSatStrobe = new THREE.Mesh(strobeGeo, strobeMat);
        activeSatStrobe.position.set(0, 0.14, 0.65);
        activeSatelliteGroup.add(activeSatStrobe);

        // Initial Orbit Position
        activeSatelliteGroup.scale.set(0.9, 0.9, 0.9);
        scene.add(activeSatelliteGroup);
    }

    // ============================================================
    // Laser Beam & Ground Concentric Shockwave Rings
    // ============================================================
    function buildLaserTargetingSystem() {
        // Red Laser Cylinder
        const beamGeo = new THREE.CylinderGeometry(0.035, 0.05, 1, 12, 1, true);
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

        // Traveling Energy Sparks along the beam
        laserSparks = [];
        for (let i = 0; i < 6; i++) {
            const sparkGeo = new THREE.SphereGeometry(0.075, 8, 8);
            const sparkMat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0,
                blending: THREE.AdditiveBlending,
            });
            const spark = new THREE.Mesh(sparkGeo, sparkMat);
            spark.visible = false;
            scene.add(spark);
            laserSparks.push(spark);
        }

        // Concentric Shockwave Rings at Target Ground Zero
        groundPingGroup = new THREE.Group();
        pingRings = [];
        for (let i = 0; i < 4; i++) {
            const ringGeo = new THREE.RingGeometry(0.08, 0.12, 32);
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
    function buildGroundMarkers() {
        groundMarkerGroup = new THREE.Group();
        earthMesh.add(groundMarkerGroup);

        // Default primary target on Texas / North America (Image 4 location)
        const primaryLoc = FINANCIAL_LOCATIONS["texas"];
        const pPos = latLonToVector3(primaryLoc.lat, primaryLoc.lon, GLOBE_RADIUS);

        // Red Ground Reticle Crosshair
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

        // Secondary Tag: Shane Woods, WA (TKV - 945K) as seen in Image 4
        const waLoc = FINANCIAL_LOCATIONS["washington"];
        const waPos = latLonToVector3(waLoc.lat, waLoc.lon, GLOBE_RADIUS);
        const waDotGeo = new THREE.SphereGeometry(0.06, 8, 8);
        const waDotMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
        const waDot = new THREE.Mesh(waDotGeo, waDotMat);
        waDot.position.copy(waPos);
        groundMarkerGroup.add(waDot);

        // Subtly illuminate financial hubs with golden pin lights
        Object.keys(FINANCIAL_LOCATIONS).forEach((key) => {
            const loc = FINANCIAL_LOCATIONS[key];
            const pos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS);
            const dotGeo = new THREE.SphereGeometry(0.035, 6, 6);
            const dotMat = new THREE.MeshBasicMaterial({
                color: key === "california" || key === "texas" ? 0xef4444 : 0xffd060,
                transparent: true,
                opacity: 0.75,
            });
            const dot = new THREE.Mesh(dotGeo, dotMat);
            dot.position.copy(pos);
            groundMarkerGroup.add(dot);
        });
    }

    // ============================================================
    // Real-Time Laser Interception Strike on Fraud Detection
    // ============================================================
    function triggerSatelliteInterception(txData) {
        if (!scene || !earthMesh) return;

        const loc = resolveLocation(txData?.location || txData?.city || "Texas");
        const isFraud = txData?.prediction === "Fraud" || (txData?.risk_score && txData.risk_score >= 42) || txData?.is_fraud;
        const txId = txData?.transaction_id || `TX-WIRE-${Math.floor(1000 + Math.random() * 9000)}`;
        const amount = Number(txData?.amount || 8450.0).toLocaleString("en-US", { style: "currency", currency: "USD" });
        const score = txData?.risk_score !== undefined ? Number(txData.risk_score).toFixed(1) : (isFraud ? "89.0" : "4.2");

        // 1. Target Vector on Earth Surface
        const localTargetPos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS);
        const worldTargetPos = localTargetPos.clone().applyEuler(earthMesh.rotation);

        // 2. Play Audio Cue
        playLaserAudio(isFraud);

        // 3. Configure Laser Colors
        const beamColor = isFraud ? 0xef4444 : 0x10b981;
        laserBeamMesh.material.color.setHex(beamColor);
        laserBeamMesh.visible = true;
        laserBeamMesh.material.opacity = 1.0;

        // Position Active Satellite directly above the target region
        currentOrbitAngle = Math.atan2(worldTargetPos.z, worldTargetPos.x) + 0.35;

        // Configure Traveling Energy Sparks
        laserSparks.forEach((spark, idx) => {
            spark.visible = true;
            spark.material.color.setHex(0xffffff);
            spark.userData = { prog: (idx * 0.18) % 1.0 };
        });

        // Configure Concentric Impact Rings
        groundPingGroup.position.copy(worldTargetPos);
        groundPingGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), worldTargetPos.clone().normalize());
        pingRings.forEach((p, idx) => {
            p.mesh.visible = true;
            p.mesh.material.color.setHex(beamColor);
            p.mesh.material.opacity = 1.0;
            p.mesh.scale.set(1, 1, 1);
            p.progress = -idx * 0.25;
        });

        // 4. Save Interception State
        activeInterception = {
            duration: 210, // ~3.5 seconds at 60 FPS
            frame: 0,
            worldTargetPos: worldTargetPos,
            localTargetPos: localTargetPos,
            locData: loc,
            txId: txId,
            amount: amount,
            score: score,
            isFraud: isFraud,
        };

        // 5. Smoothly Rotate Earth so Target Faces Viewer (Matching Image 4 Angle)
        const targetRotY = -(loc.lon * Math.PI) / 180 + Math.PI * 0.95;
        smoothRotateEarthTo(targetRotY, 1100);

        // 6. Update Image 4 Floating HUD Box
        updateImage4HudCard(loc, txId, amount, score, isFraud);

        // 7. Update Telemetry UI Displays
        updateHeaderAndDeckTelemetry(loc, txId, isFraud);
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
        const probText = isFraud ? `Chance of collision: ${score}%` : `Clearance Confirmed: 99.8%`;
        const timeStr = new Date().toLocaleTimeString();

        hud.innerHTML = `
            <div class="hud-leader-line"></div>
            <div class="hud-box-inner ${sevClass}">
                <div class="hud-top-line">
                    <span class="hud-title-txt">${probText}</span>
                    <span class="hud-status-dot ${sevClass}"></span>
                </div>
                <div class="hud-bottom-line">
                    <span>Est. time: ${timeStr}</span>
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
            const loc = FINANCIAL_LOCATIONS["texas"];
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

        // Add row to location tracking docket
        const tableBody = document.getElementById("satLocationDocketBody");
        if (tableBody) {
            const row = document.createElement("tr");
            row.className = isFraud ? "sat-docket-row-danger" : "sat-docket-row-clean";
            row.innerHTML = `
                <td><span class="docket-beacon ${isFraud ? 'danger' : 'clean'}"></span> <strong>${loc.flag} ${loc.city}</strong></td>
                <td class="font-mono">${txId}</td>
                <td class="font-mono">${loc.lat.toFixed(2)}°, ${loc.lon.toFixed(2)}°</td>
                <td><span class="status-pill-badge ${isFraud ? 'badge-sev high' : 'badge-sev low'}">${isFraud ? '🚨 FRAUD' : '✔ CLEARED'}</span></td>
                <td class="font-mono text-muted">14ms</td>
            `;
            tableBody.insertBefore(row, tableBody.firstChild);
            if (tableBody.children.length > 5) tableBody.removeChild(tableBody.lastChild);
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

        // Test Fire Button
        const btnTestFire = document.getElementById("btnTestFireSatellite");
        if (btnTestFire) {
            btnTestFire.addEventListener("click", () => {
                const sampleFrauds = [
                    { transaction_id: "WIRE-9896", location: "Texas", amount: 8450.00, risk_score: 89.0, prediction: "Fraud" },
                    { transaction_id: "WIRE-3227", location: "California", amount: 12500.00, risk_score: 96.4, prediction: "Fraud" },
                    { transaction_id: "WIRE-4812", location: "New York", amount: 4890.00, risk_score: 91.2, prediction: "Fraud" },
                    { transaction_id: "WIRE-7731", location: "London", amount: 32000.00, risk_score: 94.8, prediction: "Fraud" },
                    { transaction_id: "WIRE-5509", location: "Zurich", amount: 76000.00, risk_score: 98.1, prediction: "Fraud" },
                ];
                const picked = sampleFrauds[Math.floor(Math.random() * sampleFrauds.length)];
                triggerSatelliteInterception(picked);
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

        // 4. Animate Active Starlink Satellite Orbit
        if (activeSatelliteGroup) {
            currentOrbitAngle += 0.0035;
            const satX = Math.cos(currentOrbitAngle) * SATELLITE_ORBIT_RADIUS;
            const satZ = Math.sin(currentOrbitAngle) * SATELLITE_ORBIT_RADIUS;
            const satY = Math.sin(currentOrbitAngle) * 3.8;
            activeSatelliteGroup.position.set(satX, satY, satZ);

            // Satellite faces Earth center
            activeSatelliteGroup.lookAt(0, 0, 0);

            // Flashing navigation strobe
            strobeTick += 0.05;
            if (activeSatStrobe) {
                activeSatStrobe.material.opacity = Math.sin(strobeTick * 7) > 0.4 ? 1.0 : 0.15;
                activeSatStrobe.material.transparent = true;
            }
        }

        // 5. Active Laser Interception Lifecycle
        if (activeInterception) {
            activeInterception.frame++;

            const currentGroundTarget = activeInterception.localTargetPos
                ? activeInterception.localTargetPos.clone().applyEuler(earthMesh.rotation)
                : activeInterception.worldTargetPos;

            const satPos = activeSatelliteGroup.position.clone();

            // Connect cylinder beam between active Starlink satellite and ground zero
            const beamDist = satPos.distanceTo(currentGroundTarget);
            laserBeamMesh.scale.set(1, beamDist, 1);
            laserBeamMesh.position.copy(satPos.clone().add(currentGroundTarget).multiplyScalar(0.5));
            laserBeamMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), currentGroundTarget.clone().sub(satPos).normalize());

            // Animate Traveling Spark Packets
            laserSparks.forEach((spark) => {
                spark.userData.prog = (spark.userData.prog + 0.04) % 1.0;
                spark.position.copy(satPos.clone().lerp(currentGroundTarget, spark.userData.prog));
            });

            // Animate Concentric Shockwave Rings on Target
            groundPingGroup.position.copy(currentGroundTarget);
            pingRings.forEach((p) => {
                p.progress += 0.025;
                if (p.progress > 0) {
                    const ringProg = p.progress % 1.0;
                    p.mesh.scale.set(1 + ringProg * 4.5, 1 + ringProg * 4.5, 1);
                    p.mesh.material.opacity = Math.max(0, 1 - ringProg);
                }
            });

            // Expire laser
            if (activeInterception.frame > activeInterception.duration) {
                laserBeamMesh.visible = false;
                laserSparks.forEach((s) => (s.visible = false));
                pingRings.forEach((p) => (p.mesh.visible = false));
                activeInterception = null;
            }
        }

        // 6. Camera Modes
        if (cameraMode === "chase" && activeSatelliteGroup) {
            const chaseOffset = new THREE.Vector3(0, 2.2, 4.2).applyQuaternion(activeSatelliteGroup.quaternion);
            const desiredPos = activeSatelliteGroup.position.clone().add(chaseOffset);
            camera.position.lerp(desiredPos, 0.05);
            camera.lookAt(0, 0, 0);
        } else if (cameraMode === "target" && activeInterception) {
            const desiredPos = activeInterception.worldTargetPos.clone().multiplyScalar(2.1);
            camera.position.lerp(desiredPos, 0.04);
            camera.lookAt(activeInterception.worldTargetPos);
        } else {
            camera.lookAt(0, 0, 0);
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
        resolveLocation: resolveLocation,
        onResize: onResizeHandler,
        locations: FINANCIAL_LOCATIONS,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initSatelliteTheater);
    } else {
        setTimeout(initSatelliteTheater, 100);
    }
})();
