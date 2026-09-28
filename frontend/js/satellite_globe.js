// ============================================================
// Satellite Intelligence & Global Fraud Defense System
// 3D Orbital Earth, Satellite Tracking & Real-Time Laser Interception
// Powered by Three.js (Photorealistic Night Lights & Orbital Telemetry)
// ============================================================

(function () {
    "use strict";

    // Known Geographic Coordinates for Project Locations & Financial Centers
    const FINANCIAL_LOCATIONS = {
        "california": { lat: 36.7783, lon: -119.4179, city: "California, USA", region: "North America", flag: "🇺🇸", ipPrefix: "240.44" },
        "texas": { lat: 31.9686, lon: -99.9018, city: "Texas, USA", region: "North America", flag: "🇺🇸", ipPrefix: "28.75" },
        "new york": { lat: 40.7128, lon: -74.0060, city: "New York, USA", region: "North America", flag: "🇺🇸", ipPrefix: "198.51" },
        "florida": { lat: 27.6648, lon: -81.5158, city: "Florida, USA", region: "North America", flag: "🇺🇸", ipPrefix: "172.56" },
        "illinois": { lat: 40.6331, lon: -89.3985, city: "Chicago, USA", region: "North America", flag: "🇺🇸", ipPrefix: "192.0" },
        "washington": { lat: 47.7511, lon: -120.7401, city: "Seattle, USA", region: "North America", flag: "🇺🇸", ipPrefix: "204.79" },
        "london": { lat: 51.5074, lon: -0.1278, city: "London, UK", region: "Europe", flag: "🇬🇧", ipPrefix: "185.86" },
        "frankfurt": { lat: 50.1109, lon: 8.6821, city: "Frankfurt, Germany", region: "Europe", flag: "🇩🇪", ipPrefix: "194.12" },
        "zurich": { lat: 47.3769, lon: 8.5417, city: "Zurich, Switzerland", region: "Europe", flag: "🇨🇭", ipPrefix: "193.134" },
        "tokyo": { lat: 35.6762, lon: 139.6503, city: "Tokyo, Japan", region: "Asia-Pacific", flag: "🇯🇵", ipPrefix: "133.242" },
        "singapore": { lat: 1.3521, lon: 103.8198, city: "Singapore", region: "Asia-Pacific", flag: "🇸🇬", ipPrefix: "165.22" },
        "sydney": { lat: -33.8688, lon: 151.2093, city: "Sydney, Australia", region: "Oceania", flag: "🇦🇺", ipPrefix: "139.130" },
        "dubai": { lat: 25.2048, lon: 55.2708, city: "Dubai, UAE", region: "Middle East", flag: "🇦🇪", ipPrefix: "94.200" },
        "paris": { lat: 48.8566, lon: 2.3522, city: "Paris, France", region: "Europe", flag: "🇫🇷", ipPrefix: "195.154" },
        "sao paulo": { lat: -23.5505, lon: -46.6333, city: "São Paulo, Brazil", region: "South America", flag: "🇧🇷", ipPrefix: "177.18" },
        "mumbai": { lat: 19.0760, lon: 72.8777, city: "Mumbai, India", region: "South Asia", flag: "🇮🇳", ipPrefix: "103.21" }
    };

    // State Variables
    let scene, camera, renderer, container;
    let earthMesh, atmosphereMesh, gridMesh;
    let satelliteGroup, solarPanels = [], satStrobe;
    let laserBeamGroup, beamMesh, beamSparks = [];
    let groundPingGroup, pingRings = [];
    let locationMarkersGroup;
    let activeHudMarker = null;

    const GLOBE_RADIUS = 6.0;
    const SATELLITE_ORBIT_RADIUS = 9.6;
    let satOrbitAngle = 0;
    let satOrbitSpeed = 0.007;
    const SATELLITE_INCLINATION = 0.62; // ~35.5 degrees inclination

    let isUserDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let globeRotationSpeed = 0.0018;
    let cameraMode = "orbital"; // "orbital", "chase", "target"
    let targetCameraPosition = null;
    let targetCameraLookAt = null;

    // Laser Audio Synthesizer
    let laserAudioCtx = null;
    function playLaserSound(isFraud) {
        try {
            if (!laserAudioCtx) {
                laserAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (laserAudioCtx.state === "suspended") {
                laserAudioCtx.resume();
            }
            const now = laserAudioCtx.currentTime;
            const osc = laserAudioCtx.createOscillator();
            const gain = laserAudioCtx.createGain();
            osc.connect(gain);
            gain.connect(laserAudioCtx.destination);

            if (isFraud) {
                // High-energy laser zap descending pitch
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(1400, now);
                osc.frequency.exponentialRampToValueAtTime(180, now + 0.32);
                gain.gain.setValueAtTime(0.18, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.32);
                osc.start(now);
                osc.stop(now + 0.32);
            } else {
                // Clean frequency lock chime
                osc.type = "sine";
                osc.frequency.setValueAtTime(600, now);
                osc.frequency.exponentialRampToValueAtTime(1100, now + 0.22);
                gain.gain.setValueAtTime(0.12, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
                osc.start(now);
                osc.stop(now + 0.22);
            }
        } catch (e) {
            // Audio context silently ignored if restricted by browser policy
        }
    }

    // Convert Lat/Lon to 3D Cartesian coordinates on sphere
    function latLonToVector3(lat, lon, radius) {
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);

        const x = -(radius * Math.sin(phi) * Math.cos(theta));
        const z = radius * Math.sin(phi) * Math.sin(theta);
        const y = radius * Math.cos(phi);

        return new THREE.Vector3(x, y, z);
    }

    // Resolve location string to location data object
    function resolveLocation(locStr) {
        if (!locStr) return FINANCIAL_LOCATIONS["california"];
        const lower = String(locStr).toLowerCase().trim();
        for (const key in FINANCIAL_LOCATIONS) {
            if (lower.includes(key)) {
                return FINANCIAL_LOCATIONS[key];
            }
        }
        // Deterministic hash fallback across financial centers
        const keys = Object.keys(FINANCIAL_LOCATIONS);
        let hash = 0;
        for (let i = 0; i < lower.length; i++) {
            hash = (hash << 5) - hash + lower.charCodeAt(i);
            hash |= 0;
        }
        return FINANCIAL_LOCATIONS[keys[Math.abs(hash) % keys.length]];
    }

    // ============================================================
    // Initialize Three.js 3D Satellite Intelligence Theater
    // ============================================================
    function initSatelliteTheater() {
        container = document.getElementById("satelliteTheaterContainer");
        if (!container || typeof THREE === "undefined") return;

        const width = container.clientWidth || 900;
        const height = container.clientHeight || 560;

        // 1. Scene & Camera
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 5, 20);

        // 2. WebGL Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.25;

        // Clear existing canvas if any
        container.querySelectorAll("canvas.three-canvas").forEach((c) => c.remove());
        renderer.domElement.className = "three-canvas";
        renderer.domElement.style.width = "100%";
        renderer.domElement.style.height = "100%";
        renderer.domElement.style.display = "block";
        container.appendChild(renderer.domElement);

        // 3. Lighting (Sunlight + Ambient Earthshine + Rim Lighting)
        const ambientLight = new THREE.AmbientLight(0x1a2638, 1.4);
        scene.add(ambientLight);

        const sunLight = new THREE.DirectionalLight(0xfff6dd, 2.4);
        sunLight.position.set(20, 12, 18);
        scene.add(sunLight);

        const rimLight = new THREE.DirectionalLight(0xd4af37, 1.2);
        rimLight.position.set(-20, -10, -15);
        scene.add(rimLight);

        const cyanBacklight = new THREE.PointLight(0x38bdf8, 1.8, 40);
        cyanBacklight.position.set(-10, 8, -12);
        scene.add(cyanBacklight);

        // 4. Construct 3D Earth Globe
        buildEarthGlobe();

        // 5. Construct 3D Satellite Model & Orbit
        buildSatelliteModel();

        // 6. Laser Beam & Ground Impact Systems
        buildLaserBeamSystem();

        // 7. Pinned Financial Location Beacons
        buildLocationMarkers();

        // 8. Event Listeners for Interaction & Drag Rotate
        initInteractionEvents();

        // 9. Frequency Spectrum Waveform Visualizer
        initFrequencyWaveform();

        // 10. Start 60 FPS Render Loop
        animate();
    }

    // ============================================================
    // 3D Earth Globe Construction
    // ============================================================
    function buildEarthGlobe() {
        const textureLoader = new THREE.TextureLoader();

        // Night lights texture
        const nightTexture = textureLoader.load(
            "assets/textures/earth_night.png",
            () => renderer.render(scene, camera),
            undefined,
            () => console.warn("Using procedural fallback for Earth Night Texture")
        );

        // Day/Relief texture
        const dayTexture = textureLoader.load(
            "assets/textures/earth_day.jpg",
            () => renderer.render(scene, camera),
            undefined,
            () => console.warn("Using procedural fallback for Earth Day Texture")
        );

        // Earth Material (Glossy oceanic depth with glowing gold/emerald night city grids)
        const earthMaterial = new THREE.MeshPhongMaterial({
            map: dayTexture,
            emissiveMap: nightTexture,
            emissive: new THREE.Color(0xd4af37),
            emissiveIntensity: 1.15,
            specular: new THREE.Color(0x38bdf8),
            shininess: 35,
            bumpScale: 0.05,
        });

        const earthGeometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
        earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
        earthMesh.rotation.y = 1.2;
        scene.add(earthMesh);

        // Atmospheric Fresnel Halo (Translucent Cyan/Gold Corona)
        const atmosGeometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.045, 64, 64);
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
                    float intensity = pow(0.68 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.8);
                    gl_FragColor = vec4(0.22, 0.74, 0.97, 1.0) * intensity * 1.5;
                }
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true,
        });
        atmosphereMesh = new THREE.Mesh(atmosGeometry, atmosMaterial);
        scene.add(atmosphereMesh);

        // Outer Golden Protective Shield Ring
        const ringGeo = new THREE.RingGeometry(GLOBE_RADIUS * 1.14, GLOBE_RADIUS * 1.155, 96);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0xd4af37,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.35,
        });
        const shieldRing = new THREE.Mesh(ringGeo, ringMat);
        shieldRing.rotation.x = Math.PI / 2.3;
        scene.add(shieldRing);
    }

    // ============================================================
    // 3D Orbital Satellite Construction (Recon & Fraud Interceptor)
    // ============================================================
    function buildSatelliteModel() {
        satelliteGroup = new THREE.Group();

        // 1. Central Bus (Avionics Core)
        const busGeo = new THREE.BoxGeometry(0.7, 0.7, 1.1);
        const busMat = new THREE.MeshStandardMaterial({
            color: 0xd4af37,
            metalness: 0.88,
            roughness: 0.22,
            envMapIntensity: 1.5,
        });
        const busMesh = new THREE.Mesh(busGeo, busMat);
        satelliteGroup.add(busMesh);

        // Core gold thermal wrap banding
        const bandGeo = new THREE.BoxGeometry(0.74, 0.74, 0.3);
        const bandMat = new THREE.MeshStandardMaterial({
            color: 0x0f172a,
            metalness: 0.9,
            roughness: 0.3,
        });
        const bandMesh = new THREE.Mesh(bandGeo, bandMat);
        satelliteGroup.add(bandMesh);

        // 2. High-Gain Parabolic Communications Dish (Laser Aperture)
        const dishGeo = new THREE.CylinderGeometry(0.42, 0.08, 0.22, 24, 1, true);
        const dishMat = new THREE.MeshStandardMaterial({
            color: 0xfef08a,
            metalness: 0.95,
            roughness: 0.15,
            side: THREE.DoubleSide,
        });
        const dishMesh = new THREE.Mesh(dishGeo, dishMat);
        dishMesh.position.set(0, -0.45, 0);
        dishMesh.rotation.x = Math.PI;
        satelliteGroup.add(dishMesh);

        // Laser Emitter Hub at center of dish
        const emitterGeo = new THREE.SphereGeometry(0.12, 16, 16);
        const emitterMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const emitterMesh = new THREE.Mesh(emitterGeo, emitterMat);
        emitterMesh.position.set(0, -0.52, 0);
        satelliteGroup.add(emitterMesh);

        // 3. Solar Array Wings (Port & Starboard)
        const panelWidth = 1.6;
        const panelHeight = 0.04;
        const panelDepth = 0.75;
        const panelGeo = new THREE.BoxGeometry(panelWidth, panelHeight, panelDepth);
        const panelMat = new THREE.MeshStandardMaterial({
            color: 0x1e3a8a,
            emissive: 0x0284c7,
            emissiveIntensity: 0.35,
            metalness: 0.7,
            roughness: 0.2,
        });

        // Left Wing
        const leftWing = new THREE.Mesh(panelGeo, panelMat);
        leftWing.position.set(-1.25, 0, 0);
        satelliteGroup.add(leftWing);
        solarPanels.push(leftWing);

        // Right Wing
        const rightWing = new THREE.Mesh(panelGeo, panelMat);
        rightWing.position.set(1.25, 0, 0);
        satelliteGroup.add(rightWing);
        solarPanels.push(rightWing);

        // Solar Array Boom Connectors
        const boomGeo = new THREE.CylinderGeometry(0.04, 0.04, 2.5, 8);
        const boomMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8 });
        const boomMesh = new THREE.Mesh(boomGeo, boomMat);
        boomMesh.rotation.z = Math.PI / 2;
        satelliteGroup.add(boomMesh);

        // 4. Strobe Navigation Beacon Light
        const strobeGeo = new THREE.SphereGeometry(0.08, 12, 12);
        const strobeMat = new THREE.MeshBasicMaterial({ color: 0xff0033 });
        satStrobe = new THREE.Mesh(strobeGeo, strobeMat);
        satStrobe.position.set(0, 0.45, 0.55);
        satelliteGroup.add(satStrobe);

        // Initial Orbit Placement
        satelliteGroup.scale.set(0.85, 0.85, 0.85);
        scene.add(satelliteGroup);

        // Orbital Path Trajectory Line
        const orbitCurve = new THREE.EllipseCurve(
            0, 0,
            SATELLITE_ORBIT_RADIUS, SATELLITE_ORBIT_RADIUS,
            0, 2 * Math.PI,
            false,
            0
        );
        const points = orbitCurve.getPoints(96);
        const orbitGeo = new THREE.BufferGeometry().setFromPoints(
            points.map((p) => new THREE.Vector3(p.x, 0, p.y))
        );
        const orbitMat = new THREE.LineDashedMaterial({
            color: 0xd4af37,
            dashSize: 0.5,
            gapSize: 0.3,
            transparent: true,
            opacity: 0.45,
        });
        const orbitLine = new THREE.Line(orbitGeo, orbitMat);
        orbitLine.computeLineDistances();
        orbitLine.rotation.x = SATELLITE_INCLINATION;
        scene.add(orbitLine);
    }

    // ============================================================
    // Real-Time Laser Interception Beam & Impact System
    // ============================================================
    function buildLaserBeamSystem() {
        laserBeamGroup = new THREE.Group();
        scene.add(laserBeamGroup);

        // Core Laser Cylinder (Dynamic line from Satellite to City)
        const beamGeo = new THREE.CylinderGeometry(0.06, 0.06, 1, 12);
        const beamMat = new THREE.MeshBasicMaterial({
            color: 0xff2a5f,
            transparent: true,
            opacity: 0,
        });
        beamMesh = new THREE.Mesh(beamGeo, beamMat);
        beamMesh.visible = false;
        laserBeamGroup.add(beamMesh);

        // Traveling High-Speed Laser Energy Spark Packets
        for (let i = 0; i < 6; i++) {
            const sparkGeo = new THREE.SphereGeometry(0.12, 12, 12);
            const sparkMat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0,
            });
            const spark = new THREE.Mesh(sparkGeo, sparkMat);
            spark.visible = false;
            laserBeamGroup.add(spark);
            beamSparks.push(spark);
        }

        // Ground Concentric Radar Ping Rings (Shockwave at City Surface)
        groundPingGroup = new THREE.Group();
        scene.add(groundPingGroup);

        for (let i = 0; i < 3; i++) {
            const ringGeo = new THREE.RingGeometry(0.1, 0.18, 32);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0xef4444,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.visible = false;
            groundPingGroup.add(ring);
            pingRings.push({ mesh: ring, progress: 0 });
        }
    }

    // ============================================================
    // Pinned Location Beacons on Earth
    // ============================================================
    function buildLocationMarkers() {
        locationMarkersGroup = new THREE.Group();
        earthMesh.add(locationMarkersGroup);

        for (const key in FINANCIAL_LOCATIONS) {
            const loc = FINANCIAL_LOCATIONS[key];
            const pos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS);

            // Glowing City Surface Beacon Dot
            const dotGeo = new THREE.SphereGeometry(0.09, 12, 12);
            const dotMat = new THREE.MeshBasicMaterial({
                color: key === "california" || key === "texas" ? 0xef4444 : 0x10b981,
            });
            const dot = new THREE.Mesh(dotGeo, dotMat);
            dot.position.copy(pos);
            locationMarkersGroup.add(dot);

            // Vertical Light Needle
            const needleGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.45, 6);
            const needleMat = new THREE.MeshBasicMaterial({
                color: key === "california" || key === "texas" ? 0xef4444 : 0xd4af37,
                transparent: true,
                opacity: 0.7,
            });
            const needle = new THREE.Mesh(needleGeo, needleMat);
            needle.position.copy(pos.clone().multiplyScalar(1.035));
            needle.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pos.clone().normalize());
            locationMarkersGroup.add(needle);
        }
    }

    // ============================================================
    // Trigger Real-Time Satellite Laser Interception on Predicted Transaction
    // ============================================================
    let activeInterception = null;

    function triggerSatelliteInterception(txData) {
        if (!scene || !earthMesh) return;

        const loc = resolveLocation(txData?.location || txData?.city || "Texas");
        const isFraud = txData?.prediction === "Fraud" || (txData?.risk_score && txData.risk_score >= 42) || txData?.is_fraud;
        const txId = txData?.transaction_id || `TX-WIRE-${Math.floor(1000 + Math.random() * 9000)}`;
        const amount = Number(txData?.amount || 1508.2).toLocaleString("en-US", { style: "currency", currency: "USD" });
        const score = txData?.risk_score !== undefined ? Number(txData.risk_score).toFixed(1) : (isFraud ? "100.0" : "4.2");

        // 1. Target Surface Vector in World Space
        const localTargetPos = latLonToVector3(loc.lat, loc.lon, GLOBE_RADIUS);
        const worldTargetPos = localTargetPos.clone().applyEuler(earthMesh.rotation);

        // 2. Align Satellite Orbit Close to Target for Interception Sweep
        const satPos = satelliteGroup.position.clone();

        // 3. Audio Chirp
        playLaserSound(isFraud);

        // 4. Activate Laser Beam
        const beamColor = isFraud ? 0xef4444 : 0x10b981;
        beamMesh.material.color.setHex(beamColor);
        beamMesh.visible = true;
        beamMesh.material.opacity = 1.0;

        // Position & Orient Cylinder Beam between Satellite and Ground Target
        const beamLength = satPos.distanceTo(worldTargetPos);
        beamMesh.scale.set(1, beamLength, 1);
        beamMesh.position.copy(satPos.clone().add(worldTargetPos).multiplyScalar(0.5));
        beamMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), worldTargetPos.clone().sub(satPos).normalize());

        // Setup Fast Energy Sparks traveling along the beam
        beamSparks.forEach((spark, idx) => {
            spark.visible = true;
            spark.material.color.setHex(0xffffff);
            spark.material.opacity = 1;
            spark.userData = {
                prog: (idx * 0.16) % 1.0,
                startPos: satPos.clone(),
                endPos: worldTargetPos.clone(),
            };
        });

        // Setup Concentric Shockwave Rings on Target Surface
        groundPingGroup.position.copy(worldTargetPos);
        groundPingGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), worldTargetPos.clone().normalize());

        pingRings.forEach((p, idx) => {
            p.mesh.visible = true;
            p.mesh.material.color.setHex(beamColor);
            p.mesh.material.opacity = 1.0;
            p.mesh.scale.set(1, 1, 1);
            p.progress = -idx * 0.28;
        });

        // 5. Update State
        activeInterception = {
            duration: 180, // ~3 seconds at 60 FPS
            frame: 0,
            worldTargetPos: worldTargetPos,
            locData: loc,
            txId: txId,
            amount: amount,
            score: score,
            isFraud: isFraud,
        };

        // 6. Smoothly Steer Globe toward Target so it faces viewer
        const targetRotY = -(loc.lon * Math.PI) / 180 + Math.PI;
        globeRotationSpeed = 0; // Pause auto-spin during targeted beam
        new TWEEN_SIMPLE(earthMesh.rotation, { y: targetRotY }, 1200, () => {
            globeRotationSpeed = 0.0018;
        });

        // 7. Update HUD Display Banner
        updateHudBeaconCard(loc, txId, amount, score, isFraud);

        // 8. Update Satellite Telemetry Deck
        updateTelemetryDeck(loc, txId, isFraud);

        // 9. Spike Frequency Seismograph
        triggerWaveformSpike();
    }

    // Minimal Tween helper
    function TWEEN_SIMPLE(target, dest, durationMs, onComplete) {
        const startY = target.y;
        const endY = dest.y;
        const startTime = performance.now();

        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / durationMs, 1.0);
            const ease = 0.5 - Math.cos(progress * Math.PI) / 2;
            target.y = startY + (endY - startY) * ease;
            if (progress < 1.0) {
                requestAnimationFrame(step);
            } else if (onComplete) {
                onComplete();
            }
        }
        requestAnimationFrame(step);
    }

    // ============================================================
    // Live Floating HUD Target Card
    // ============================================================
    function updateHudBeaconCard(loc, txId, amount, score, isFraud) {
        let hud = document.getElementById("satelliteTargetHudCard");
        if (!hud) {
            hud = document.createElement("div");
            hud.id = "satelliteTargetHudCard";
            hud.className = "satellite-hud-card";
            container.appendChild(hud);
        }

        const sevClass = isFraud ? "danger" : "cleared";
        const statusText = isFraud ? "● FRAUD INTERCEPTED" : "✔ SETTLED APPROVED";

        hud.innerHTML = `
            <div class="hud-card-header">
                <span class="hud-sat-beacon ${sevClass}"></span>
                <strong>SATELLITE DOWNLINK // LOCK</strong>
                <span class="hud-pill ${sevClass}">${statusText}</span>
            </div>
            <div class="hud-card-body">
                <div class="hud-row">
                    <span class="hud-lbl">TARGET TX:</span>
                    <strong class="font-mono text-white">${txId}</strong>
                </div>
                <div class="hud-row">
                    <span class="hud-lbl">COORDINATE:</span>
                    <strong class="font-mono">${loc.flag} ${loc.city}</strong>
                </div>
                <div class="hud-row">
                    <span class="hud-lbl">CLEARING SUM:</span>
                    <strong class="font-mono text-yellow">${amount}</strong>
                </div>
                <div class="hud-row">
                    <span class="hud-lbl">ML PROBABILITY:</span>
                    <strong class="font-mono ${isFraud ? 'text-danger' : 'text-green'}">Score: ${score}</strong>
                </div>
            </div>
            <div class="hud-card-footer">
                <span>BEAM LATENCY: 12ms</span>
                <span>UPLINK: SAT-DEFENSE-04</span>
            </div>
        `;

        hud.style.display = "block";
        hud.classList.remove("fade-out");

        clearTimeout(hud._timer);
        hud._timer = setTimeout(() => {
            hud.classList.add("fade-out");
            setTimeout(() => { hud.style.display = "none"; }, 500);
        }, 4500);
    }

    // Position HUD card in 2D Screen Space
    function updateHudCardScreenPosition() {
        const hud = document.getElementById("satelliteTargetHudCard");
        if (!hud || hud.style.display === "none" || !activeInterception) return;

        const targetPos = activeInterception.worldTargetPos.clone();
        targetPos.project(camera);

        // Check if behind the Earth sphere
        const isBehind = targetPos.z > 1.0 || activeInterception.worldTargetPos.clone().normalize().dot(camera.position.clone().normalize()) < -0.15;
        if (isBehind) {
            hud.style.opacity = "0.2";
            return;
        } else {
            hud.style.opacity = "1";
        }

        const widthHalf = (container.clientWidth || 900) / 2;
        const heightHalf = (container.clientHeight || 560) / 2;

        const x = targetPos.x * widthHalf + widthHalf;
        const y = -(targetPos.y * heightHalf) + heightHalf;

        hud.style.left = `${Math.min(Math.max(x + 20, 20), container.clientWidth - 260)}px`;
        hud.style.top = `${Math.min(Math.max(y - 60, 20), container.clientHeight - 160)}px`;
    }

    // ============================================================
    // Satellite Telemetry Deck Update
    // ============================================================
    function updateTelemetryDeck(loc, txId, isFraud) {
        const satStatusPill = document.getElementById("satStatusPill");
        const satTargetLocPill = document.getElementById("satTargetLocPill");
        const satSpeedVal = document.getElementById("satSpeedVal");
        const satBeamPowerVal = document.getElementById("satBeamPowerVal");

        if (satStatusPill) {
            satStatusPill.textContent = isFraud ? "INTERCEPTING THREAT" : "TARGET CLEAR";
            satStatusPill.className = `sat-telemetry-badge ${isFraud ? 'danger' : 'success'}`;
        }
        if (satTargetLocPill) {
            satTargetLocPill.textContent = `${loc.city} (${loc.lat.toFixed(1)}°, ${loc.lon.toFixed(1)}°)`;
        }
        if (satSpeedVal) {
            satSpeedVal.textContent = "27,420 km/h";
        }
        if (satBeamPowerVal) {
            satBeamPowerVal.textContent = "99.8% Nominal";
        }

        // Add or update row in location tracking docket
        updateLocationDocketTable(loc, txId, isFraud);
    }

    function updateLocationDocketTable(loc, txId, isFraud) {
        const tableBody = document.getElementById("satLocationDocketBody");
        if (!tableBody) return;

        const row = document.createElement("tr");
        row.className = isFraud ? "sat-docket-row-danger" : "sat-docket-row-clean";
        row.innerHTML = `
            <td><span class="docket-beacon ${isFraud ? 'danger' : 'clean'}"></span> <strong>${loc.flag} ${loc.city}</strong></td>
            <td class="font-mono">${txId}</td>
            <td class="font-mono">${loc.lat.toFixed(2)}°, ${loc.lon.toFixed(2)}°</td>
            <td><span class="status-pill-badge ${isFraud ? 'badge-sev high' : 'badge-sev low'}">${isFraud ? '🚨 FRAUD' : '✔ CLEARED'}</span></td>
            <td class="font-mono text-muted">12ms</td>
        `;

        tableBody.insertBefore(row, tableBody.firstChild);
        if (tableBody.children.length > 5) {
            tableBody.removeChild(tableBody.lastChild);
        }
    }

    // ============================================================
    // Real-Time Frequency Waveform Canvas (Image 3 & 4 Style)
    // ============================================================
    let waveCanvas, waveCtx;
    let waveTick = 0;
    let isSpiking = false;
    let spikeDecay = 0;

    function initFrequencyWaveform() {
        waveCanvas = document.getElementById("satWaveformCanvas");
        if (!waveCanvas) return;
        waveCtx = waveCanvas.getContext("2d");
        waveCanvas.width = waveCanvas.clientWidth || 320;
        waveCanvas.height = waveCanvas.clientHeight || 75;
    }

    function triggerWaveformSpike() {
        isSpiking = true;
        spikeDecay = 1.0;
    }

    function renderWaveform() {
        if (!waveCanvas || !waveCtx) return;
        const w = waveCanvas.width;
        const h = waveCanvas.height;
        const cy = h / 2;

        waveCtx.clearRect(0, 0, w, h);
        waveTick += 0.08;

        if (spikeDecay > 0) {
            spikeDecay *= 0.96;
            if (spikeDecay < 0.01) spikeDecay = 0;
        }

        // Waveform 1: Primary Cyan/Teal carrier wave
        waveCtx.beginPath();
        waveCtx.strokeStyle = spikeDecay > 0.3 ? "#ef4444" : "#38bdf8";
        waveCtx.lineWidth = 1.8;

        for (let x = 0; x < w; x++) {
            const freq = 0.045;
            const amp = (12 + spikeDecay * 22) * Math.sin(x * 0.02 + waveTick);
            const noise = Math.sin(x * freq + waveTick * 2) * (5 + spikeDecay * 15);
            const y = cy + Math.sin(x * freq + waveTick) * amp + noise;
            if (x === 0) waveCtx.moveTo(x, y);
            else waveCtx.lineTo(x, y);
        }
        waveCtx.stroke();

        // Waveform 2: Ambient Gold carrier
        waveCtx.beginPath();
        waveCtx.strokeStyle = "rgba(212, 175, 55, 0.4)";
        waveCtx.lineWidth = 1.0;
        for (let x = 0; x < w; x += 2) {
            const y = cy + Math.sin(x * 0.035 - waveTick * 1.5) * 8;
            if (x === 0) waveCtx.moveTo(x, y);
            else waveCtx.lineTo(x, y);
        }
        waveCtx.stroke();
    }

    // ============================================================
    // Drag, Zoom & Camera Interactions
    // ============================================================
    function initInteractionEvents() {
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

            // Clamp vertical tilt
            earthMesh.rotation.x = Math.max(-Math.PI / 4, Math.min(Math.PI / 4, earthMesh.rotation.x));
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        // Mouse Wheel Zoom
        container.addEventListener("wheel", (e) => {
            e.preventDefault();
            camera.position.z += e.deltaY * 0.015;
            camera.position.z = Math.max(10, Math.min(32, camera.position.z));
        }, { passive: false });

        // Camera Mode Selector Buttons
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

        // Test Fire Button
        const btnTestFireSatellite = document.getElementById("btnTestFireSatellite");
        if (btnTestFireSatellite) {
            btnTestFireSatellite.addEventListener("click", () => {
                const sampleFrauds = [
                    { transaction_id: "WIRE-9896", location: "Texas", amount: 1508.20, risk_score: 100.0, prediction: "Fraud" },
                    { transaction_id: "WIRE-3227", location: "California", amount: 91.62, risk_score: 100.0, prediction: "Fraud" },
                    { transaction_id: "WIRE-4812", location: "New York", amount: 4890.00, risk_score: 97.4, prediction: "Fraud" },
                    { transaction_id: "WIRE-7731", location: "London", amount: 12500.00, risk_score: 94.8, prediction: "Fraud" },
                    { transaction_id: "WIRE-5509", location: "Zurich", amount: 82000.00, risk_score: 99.1, prediction: "Fraud" },
                ];
                const picked = sampleFrauds[Math.floor(Math.random() * sampleFrauds.length)];
                triggerSatelliteInterception(picked);
            });
        }

        // Window Resize
        window.addEventListener("resize", () => {
            if (!container || !renderer || !camera) return;
            const w = container.clientWidth || 900;
            const h = container.clientHeight || 560;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
            if (waveCanvas) {
                waveCanvas.width = waveCanvas.clientWidth || 320;
                waveCanvas.height = waveCanvas.clientHeight || 75;
            }
        });
    }

    function setCameraModeActive(activeBtn) {
        document.querySelectorAll(".btn-sat-cam").forEach((b) => b.classList.remove("active"));
        if (activeBtn) activeBtn.classList.add("active");
    }

    // ============================================================
    // Main 60 FPS Render & Animation Loop
    // ============================================================
    let strobeTimer = 0;

    function animate() {
        requestAnimationFrame(animate);

        // 1. Slow Continuous Earth Spin (Live Video Feel)
        if (earthMesh && !isUserDragging) {
            earthMesh.rotation.y += globeRotationSpeed;
        }

        // 2. Satellite Orbit Dynamics
        if (satelliteGroup) {
            satOrbitAngle += satOrbitSpeed;

            const orbitX = Math.cos(satOrbitAngle) * SATELLITE_ORBIT_RADIUS;
            const orbitZ = Math.sin(satOrbitAngle) * SATELLITE_ORBIT_RADIUS;
            const orbitY = Math.sin(satOrbitAngle) * Math.sin(SATELLITE_INCLINATION) * 4.5;

            satelliteGroup.position.set(orbitX, orbitY, orbitZ);

            // Orient satellite to always point communications dish toward Earth center
            satelliteGroup.lookAt(0, 0, 0);

            // Strobe beacon flash
            strobeTimer += 0.05;
            if (satStrobe) {
                satStrobe.material.opacity = Math.sin(strobeTimer * 6) > 0.4 ? 1.0 : 0.1;
                satStrobe.material.transparent = true;
            }
        }

        // 3. Laser Beam Active Frame Lifecycle
        if (activeInterception) {
            activeInterception.frame++;

            // Ground Target Vector refreshed as Earth rotates
            const currentWorldTarget = activeInterception.localTargetPos
                ? activeInterception.localTargetPos.clone().applyEuler(earthMesh.rotation)
                : activeInterception.worldTargetPos;

            const satPos = satelliteGroup.position.clone();

            // Keep beam connected between moving satellite and rotating target
            const beamLength = satPos.distanceTo(currentWorldTarget);
            beamMesh.scale.set(1, beamLength, 1);
            beamMesh.position.copy(satPos.clone().add(currentWorldTarget).multiplyScalar(0.5));
            beamMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), currentWorldTarget.clone().sub(satPos).normalize());

            // Update Traveling Energy Spark Packets
            beamSparks.forEach((spark) => {
                spark.userData.prog = (spark.userData.prog + 0.04) % 1.0;
                spark.position.copy(satPos.clone().lerp(currentWorldTarget, spark.userData.prog));
            });

            // Update Concentric Radar Ping Rings at Ground Zero
            groundPingGroup.position.copy(currentWorldTarget);
            pingRings.forEach((p) => {
                p.progress += 0.025;
                if (p.progress > 0) {
                    const ringProg = p.progress % 1.0;
                    p.mesh.scale.set(1 + ringProg * 4.5, 1 + ringProg * 4.5, 1);
                    p.mesh.material.opacity = Math.max(0, 1 - ringProg);
                }
            });

            // End of beam duration
            if (activeInterception.frame > activeInterception.duration) {
                beamMesh.visible = false;
                beamSparks.forEach((s) => (s.visible = false));
                pingRings.forEach((p) => (p.mesh.visible = false));
                activeInterception = null;
            }
        }

        // 4. Camera Modes
        if (cameraMode === "chase" && satelliteGroup) {
            const chaseOffset = new THREE.Vector3(0, 2.5, 4.5).applyQuaternion(satelliteGroup.quaternion);
            const desiredPos = satelliteGroup.position.clone().add(chaseOffset);
            camera.position.lerp(desiredPos, 0.05);
            camera.lookAt(0, 0, 0);
        } else if (cameraMode === "target" && activeInterception) {
            const desiredPos = activeInterception.worldTargetPos.clone().multiplyScalar(2.2);
            camera.position.lerp(desiredPos, 0.04);
            camera.lookAt(activeInterception.worldTargetPos);
        } else {
            // Default Orbital Overview
            camera.lookAt(0, 0, 0);
        }

        // 5. Update HUD Card Coordinates
        updateHudCardScreenPosition();

        // 6. Render Frequency Waveform Canvas
        renderWaveform();

        // 7. Render Three.js Scene
        renderer.render(scene, camera);
    }

    // ============================================================
    // Global Public API Export
    // ============================================================
    window.SatelliteDefense = {
        init: initSatelliteTheater,
        triggerInterception: triggerSatelliteInterception,
        resolveLocation: resolveLocation,
        locations: FINANCIAL_LOCATIONS,
    };

    // Auto-init when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initSatelliteTheater);
    } else {
        setTimeout(initSatelliteTheater, 100);
    }
})();
