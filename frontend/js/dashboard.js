// ============================================================
// Intelligence Fraud Shield
// Apex Global Banking - Core Operations & Fraud Engine Controller
// ============================================================

(function () {
    "use strict";

    // Global Terminal State
    let cachedAlerts = [];
    let currentTab = "cockpit";
    let liveFeedInterval = null;
    let liveFeedRunning = false;
    let liveStreamTotal = 0;
    let liveStreamFrauds = 0;
    let liveStreamClean = 0;
    let liveStreamSavedCapital = 0;
    let soundEnabled = true;

    // Web Audio Synthesizer for Institutional Audio Cues
    let audioCtx = null;
    function playBeep(type) {
        if (!soundEnabled) return;
        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === "suspended") {
                audioCtx.resume();
            }
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);

            if (type === "alert") {
                // High alert double-ping for intercepted fraud
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(880, audioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.18);
                gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.18);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.18);
            } else if (type === "clean") {
                // Soft chime for approved payment
                osc.type = "sine";
                osc.frequency.setValueAtTime(523.25, audioCtx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(659.25, audioCtx.currentTime + 0.12);
                gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.12);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.12);
            }
        } catch (e) {
            // Audio context silently ignored if restricted by browser
        }
    }

    // ============================================================
    // Motion Graphics Background Engine (Cyber-Financial Stream)
    // ============================================================
    function initMotionCanvas() {
        const canvas = document.getElementById("cyberMotionCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        let width = (canvas.width = window.innerWidth);
        let height = (canvas.height = window.innerHeight);

        let mouseX = width / 2;
        let mouseY = height / 2;
        let targetMouseX = mouseX;
        let targetMouseY = mouseY;

        window.addEventListener("mousemove", (e) => {
            targetMouseX = e.clientX;
            targetMouseY = e.clientY;
        });

        window.addEventListener("resize", () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
            initNodes();
        });

        // Financial Node Network
        const NODE_COUNT = Math.min(Math.floor((width * height) / 28000), 55);
        let nodes = [];

        function initNodes() {
            nodes = [];
            for (let i = 0; i < NODE_COUNT; i++) {
                const isThreat = Math.random() < 0.18;
                nodes.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: (Math.random() - 0.5) * 0.45,
                    vy: (Math.random() - 0.5) * 0.45,
                    radius: isThreat ? Math.random() * 2 + 2.5 : Math.random() * 1.5 + 1.2,
                    isThreat: isThreat,
                    pulse: Math.random() * Math.PI * 2,
                    pulseSpeed: Math.random() * 0.04 + 0.02,
                });
            }
        }
        initNodes();

        // Liquidity Beziers (Flowing Wire Waves)
        const WAVES = [
            { yRatio: 0.25, speed: 0.0008, amp: 55, sparkOffset: 0, type: "gold" },
            { yRatio: 0.52, speed: 0.0012, amp: 85, sparkOffset: 0.4, type: "crimson" },
            { yRatio: 0.78, speed: 0.0009, amp: 65, sparkOffset: 0.7, type: "emerald" },
        ];

        let tick = 0;

        function render() {
            tick++;
            // Smooth mouse lag
            mouseX += (targetMouseX - mouseX) * 0.04;
            mouseY += (targetMouseY - mouseY) * 0.04;
            const mouseOffsetX = (mouseX / width - 0.5) * 25;
            const mouseOffsetY = (mouseY / height - 0.5) * 25;

            const theme = document.body.getAttribute("data-theme") || "gold";

            // Theme-calibrated color maps
            let bgFill = "#070a12";
            let orb1Color = "rgba(212, 175, 55, 0.14)";
            let orb2Color = "rgba(239, 68, 68, 0.10)";
            let threatNodeColor = "#ef4444";
            let threatShadow = "rgba(239, 68, 68, 0.8)";
            let standardNodeColor = "#d4af37";
            let standardShadow = "rgba(212, 175, 55, 0.65)";
            let waveGoldColor = "rgba(212, 175, 55, 0.16)";
            let waveCrimsonColor = "rgba(239, 68, 68, 0.14)";
            let waveEmeraldColor = "rgba(16, 185, 129, 0.14)";

            if (theme === "cyber") {
                bgFill = "#030408";
                orb1Color = "rgba(255, 0, 85, 0.16)";
                orb2Color = "rgba(0, 255, 136, 0.12)";
                threatNodeColor = "#ff0033";
                threatShadow = "rgba(255, 0, 51, 0.85)";
                standardNodeColor = "#00ff88";
                standardShadow = "rgba(0, 255, 136, 0.7)";
                waveGoldColor = "rgba(255, 0, 85, 0.18)";
                waveCrimsonColor = "rgba(255, 0, 51, 0.18)";
                waveEmeraldColor = "rgba(0, 255, 136, 0.16)";
            } else if (theme === "obsidian") {
                bgFill = "#040406";
                orb1Color = "rgba(255, 255, 255, 0.10)";
                orb2Color = "rgba(255, 51, 75, 0.12)";
                threatNodeColor = "#ff334b";
                threatShadow = "rgba(255, 51, 75, 0.8)";
                standardNodeColor = "#e2e8f0";
                standardShadow = "rgba(255, 255, 255, 0.6)";
                waveGoldColor = "rgba(226, 232, 240, 0.14)";
                waveCrimsonColor = "rgba(255, 51, 75, 0.14)";
                waveEmeraldColor = "rgba(16, 185, 129, 0.12)";
            } else if (theme === "light") {
                bgFill = "#f4f6fa";
                orb1Color = "rgba(184, 134, 11, 0.08)";
                orb2Color = "rgba(220, 38, 38, 0.06)";
                threatNodeColor = "#dc2626";
                threatShadow = "rgba(220, 38, 38, 0.4)";
                standardNodeColor = "#b8860b";
                standardShadow = "rgba(184, 134, 11, 0.4)";
                waveGoldColor = "rgba(184, 134, 11, 0.12)";
                waveCrimsonColor = "rgba(220, 38, 38, 0.10)";
                waveEmeraldColor = "rgba(5, 150, 105, 0.10)";
            }

            // 0. Space background with theme clear
            ctx.fillStyle = bgFill;
            ctx.fillRect(0, 0, width, height);

            // 1. Ambient Breathing Radial Orbs
            const orb1X = width * 0.2 + Math.sin(tick * 0.008) * 60 + mouseOffsetX;
            const orb1Y = height * 0.3 + Math.cos(tick * 0.007) * 40 + mouseOffsetY;
            const grad1 = ctx.createRadialGradient(orb1X, orb1Y, 10, orb1X, orb1Y, width * 0.45);
            grad1.addColorStop(0, orb1Color);
            grad1.addColorStop(0.5, "rgba(0, 0, 0, 0.05)");
            grad1.addColorStop(1, "transparent");
            ctx.fillStyle = grad1;
            ctx.fillRect(0, 0, width, height);

            const orb2X = width * 0.8 + Math.cos(tick * 0.009) * 50 - mouseOffsetX;
            const orb2Y = height * 0.7 + Math.sin(tick * 0.006) * 50 - mouseOffsetY;
            const grad2 = ctx.createRadialGradient(orb2X, orb2Y, 10, orb2X, orb2Y, width * 0.38);
            grad2.addColorStop(0, orb2Color);
            grad2.addColorStop(0.6, "transparent");
            ctx.fillStyle = grad2;
            ctx.fillRect(0, 0, width, height);

            // 2. Flowing Liquidity Wire Waves with Traveling Sparks
            WAVES.forEach((wave) => {
                ctx.beginPath();
                const baseY = height * wave.yRatio;
                ctx.moveTo(0, baseY);
                for (let x = 0; x <= width; x += 40) {
                    const y = baseY + Math.sin(x * 0.003 + tick * wave.speed * 60) * wave.amp + Math.cos(x * 0.002 + tick * 0.01) * 20;
                    ctx.lineTo(x, y);
                }
                const waveStroke = wave.type === "gold" ? waveGoldColor : (wave.type === "crimson" ? waveCrimsonColor : waveEmeraldColor);
                ctx.strokeStyle = waveStroke;
                ctx.lineWidth = 1.8;
                ctx.stroke();

                // Spark particle traveling along the wave
                const sparkProg = (tick * 0.002 + wave.sparkOffset) % 1;
                const sparkX = sparkProg * width;
                const sparkY = baseY + Math.sin(sparkX * 0.003 + tick * wave.speed * 60) * wave.amp + Math.cos(sparkX * 0.002 + tick * 0.01) * 20;
                ctx.beginPath();
                ctx.arc(sparkX, sparkY, 3, 0, Math.PI * 2);
                ctx.fillStyle = wave.type === "crimson" ? threatNodeColor : (wave.type === "emerald" ? "#10b981" : standardNodeColor);
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0;
            });

            // 3. Financial Node Constellation
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                node.x += node.vx;
                node.y += node.vy;
                node.pulse += node.pulseSpeed;

                if (node.x < 0) node.x = width;
                if (node.x > width) node.x = 0;
                if (node.y < 0) node.y = height;
                if (node.y > height) node.y = 0;

                // Connecting filaments
                for (let j = i + 1; j < nodes.length; j++) {
                    const other = nodes[j];
                    const dx = other.x - node.x;
                    const dy = other.y - node.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 130) {
                        const alpha = (1 - dist / 130) * 0.16;
                        ctx.beginPath();
                        ctx.moveTo(node.x, node.y);
                        ctx.lineTo(other.x, other.y);
                        ctx.strokeStyle = (node.isThreat || other.isThreat)
                            ? `rgba(239, 68, 68, ${alpha * 1.5})`
                            : (theme === "cyber" ? `rgba(0, 255, 136, ${alpha})` : `rgba(212, 175, 55, ${alpha})`);
                        ctx.lineWidth = 0.8;
                        ctx.stroke();
                    }
                }

                // Render node
                ctx.beginPath();
                const currentRadius = node.radius + Math.sin(node.pulse) * 0.6;
                ctx.arc(node.x, node.y, Math.max(currentRadius, 1), 0, Math.PI * 2);

                if (node.isThreat) {
                    ctx.fillStyle = threatNodeColor;
                    ctx.shadowColor = threatShadow;
                    ctx.shadowBlur = 12;
                } else {
                    ctx.fillStyle = standardNodeColor;
                    ctx.shadowColor = standardShadow;
                    ctx.shadowBlur = 8;
                }
                ctx.fill();
                ctx.shadowBlur = 0;
            }

            requestAnimationFrame(render);
        }

        render();
    }

    // DOM Elements - Shell & Navigation
    const authPortalSection = document.getElementById("authPortalSection");
    const bankingWorkspace = document.getElementById("bankingWorkspace");
    const bankNavHub = document.getElementById("bankNavHub");
    const operatorBadge = document.getElementById("operatorBadge");
    const operatorName = document.getElementById("operatorName");
    const operatorRole = document.getElementById("operatorRole");
    const logoutBtn = document.getElementById("logoutBtn");
    const systemRefreshBtn = document.getElementById("systemRefreshBtn");
    const soundToggleBtn = document.getElementById("soundToggleBtn");
    const soundIcon = document.getElementById("soundIcon");
    const liveClock = document.getElementById("liveClock");
    const latencyPill = document.getElementById("latencyPill");
    const activeAlertBadge = document.getElementById("activeAlertBadge");

    // Header Status Pills
    const apiStatusPill = document.getElementById("apiStatusPill");
    const dbStatusPill = document.getElementById("dbStatusPill");
    const cutoffStatusPill = document.getElementById("cutoffStatusPill");

    // Login Form Elements
    const bankingAuthForm = document.getElementById("bankingAuthForm");
    const authUsername = document.getElementById("authUsername");
    const authPassword = document.getElementById("authPassword");
    const authSubmitBtn = document.getElementById("authSubmitBtn");
    const authErrorMessage = document.getElementById("authErrorMessage");
    const btnAdminQuickLogin = document.getElementById("btnAdminQuickLogin");
    const btnAnalystQuickLogin = document.getElementById("btnAnalystQuickLogin");

    // Cockpit Elements (Tab 1)
    const lastSyncTime = document.getElementById("lastSyncTime");
    const statTotalAlerts = document.getElementById("statTotalAlerts");
    const statNewAlerts = document.getElementById("statNewAlerts");
    const statUnderReview = document.getElementById("statUnderReview");
    const statResolved = document.getElementById("statResolved");
    const barHighCount = document.getElementById("barHighCount");
    const barMedCount = document.getElementById("barMedCount");
    const barLowCount = document.getElementById("barLowCount");
    const barHighFill = document.getElementById("barHighFill");
    const barMedFill = document.getElementById("barMedFill");
    const barLowFill = document.getElementById("barLowFill");
    const cockpitDbState = document.getElementById("cockpitDbState");
    const cockpitMiniAlerts = document.getElementById("cockpitMiniAlerts");

    // Simulator Elements (Tab 2)
    const wireTransferForm = document.getElementById("wireTransferForm");
    const btnGenNewWireId = document.getElementById("btnGenNewWireId");
    const btnPresetAttack = document.getElementById("btnPresetAttack");
    const btnPresetAnomaly = document.getElementById("btnPresetAnomaly");
    const btnPresetClean = document.getElementById("btnPresetClean");
    const wireDeviation = document.getElementById("wireDeviation");
    const wireDeviationBadge = document.getElementById("wireDeviationBadge");
    const wireThreshold = document.getElementById("wireThreshold");
    const wireThresholdBadge = document.getElementById("wireThresholdBadge");
    const btnExecuteScoring = document.getElementById("btnExecuteScoring");

    // Simulator HUD Elements
    const hudTimestamp = document.getElementById("hudTimestamp");
    const hudEmptyState = document.getElementById("hudEmptyState");
    const hudLiveScored = document.getElementById("hudLiveScored");
    const hudGaugeFill = document.getElementById("hudGaugeFill");
    const hudRiskScore = document.getElementById("hudRiskScore");
    const hudVerdictBanner = document.getElementById("hudVerdictBanner");
    const hudVerdictIcon = document.getElementById("hudVerdictIcon");
    const hudVerdictText = document.getElementById("hudVerdictText");
    const hudProbVal = document.getElementById("hudProbVal");
    const hudTierVal = document.getElementById("hudTierVal");
    const hudCutoffVal = document.getElementById("hudCutoffVal");
    const hudDbSyncVal = document.getElementById("hudDbSyncVal");
    const hudXaiList = document.getElementById("hudXaiList");

    // Live Stream Elements (Tab 3)
    const btnToggleLiveFeed = document.getElementById("btnToggleLiveFeed");
    const liveFeedIcon = document.getElementById("liveFeedIcon");
    const liveFeedText = document.getElementById("liveFeedText");
    const btnClearLiveFeed = document.getElementById("btnClearLiveFeed");
    const streamCount = document.getElementById("streamCount");
    const streamFraudCount = document.getElementById("streamFraudCount");
    const streamCleanCount = document.getElementById("streamCleanCount");
    const streamSavedCapital = document.getElementById("streamSavedCapital");
    const streamTableBody = document.getElementById("streamTableBody");

    // Batch Elements (Tab 4)
    const btnExecute197Batch = document.getElementById("btnExecute197Batch");
    const batchCustomFileInput = document.getElementById("batchCustomFileInput");
    const batchCustomFileLabel = document.getElementById("batchCustomFileLabel");
    const batchLoadingIndicator = document.getElementById("batchLoadingIndicator");
    const batchResultsHub = document.getElementById("batchResultsHub");
    const batchLatencyBadge = document.getElementById("batchLatencyBadge");
    const bmTotal = document.getElementById("bmTotal");
    const bmSuspicious = document.getElementById("bmSuspicious");
    const bmNormal = document.getElementById("bmNormal");
    const bmRate = document.getElementById("bmRate");
    const bmAvgRisk = document.getElementById("bmAvgRisk");
    const batchPreviewTableBody = document.getElementById("batchPreviewTableBody");

    // Triage Desk Elements (Tab 5)
    const triageStatusFilter = document.getElementById("triageStatusFilter");
    const triageRiskFilter = document.getElementById("triageRiskFilter");
    const triageSearchInput = document.getElementById("triageSearchInput");
    const triageLoadingMsg = document.getElementById("triageLoadingMsg");
    const triageErrorMsg = document.getElementById("triageErrorMsg");
    const triageTableBody = document.getElementById("triageTableBody");
    const triageCountLabel = document.getElementById("triageCountLabel");
    const btnExportTriageCsv = document.getElementById("btnExportTriageCsv");
    const btnReloadTriage = document.getElementById("btnReloadTriage");

    // Toast Hub
    const toastNotification = document.getElementById("toastNotification");
    const toastIcon = document.getElementById("toastIcon");
    const toastMessage = document.getElementById("toastMessage");

    // ============================================================
    // Toast Notification System
    // ============================================================
    function showToast(message, type = "success") {
        if (!toastNotification || !toastMessage) return;
        toastMessage.textContent = message;
        toastIcon.textContent = type === "error" ? "🚨" : type === "warning" ? "⚠️" : "🔔";
        toastNotification.className = `toast-hub ${type}`;
        toastNotification.classList.remove("hidden");

        setTimeout(() => {
            toastNotification.classList.add("hidden");
        }, 3800);
    }

    // ============================================================
    // Real-Time Clock & Network Latency Simulator
    // ============================================================
    function updateClock() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        if (liveClock) liveClock.textContent = `${timeStr} (UTC+5:30)`;

        // Periodic jitter for realistic latency telemetry
        if (latencyPill && Math.random() < 0.15) {
            const jitter = Math.floor(16 + Math.random() * 8);
            latencyPill.textContent = `${jitter}ms`;
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // ============================================================
    // Tab Navigation Routing
    // ============================================================
    function switchTab(tabName) {
        currentTab = tabName;
        const tabs = document.querySelectorAll(".bank-nav-tab");
        const views = document.querySelectorAll(".workspace-view");

        tabs.forEach((tab) => {
            if (tab.getAttribute("data-tab") === tabName) {
                tab.classList.add("active");
            } else {
                tab.classList.remove("active");
            }
        });

        views.forEach((view) => {
            if (view.id === `view-${tabName}`) {
                view.classList.add("active");
            } else {
                view.classList.remove("active");
            }
        });

        if (tabName === "triage") {
            loadTriageCases();
        } else if (tabName === "cockpit") {
            loadCockpitData();
            if (window.SatelliteDefense && typeof window.SatelliteDefense.init === "function") {
                setTimeout(() => {
                    window.dispatchEvent(new Event("resize"));
                }, 80);
            }
        }
    }

    // ============================================================
    // Authentication State Management
    // ============================================================
    function setOperatorAuthenticated(username, role) {
        authPortalSection.classList.add("hidden");
        bankingWorkspace.classList.remove("hidden");
        bankNavHub.classList.remove("hidden");
        operatorBadge.classList.remove("hidden");
        logoutBtn.classList.remove("hidden");

        if (operatorName) operatorName.textContent = username.toUpperCase();
        if (operatorRole) operatorRole.textContent = role || "Risk Operations Officer";

        loadCockpitData();
    }

    function setOperatorLoggedOut() {
        authPortalSection.classList.remove("hidden");
        bankingWorkspace.classList.add("hidden");
        bankNavHub.classList.add("hidden");
        operatorBadge.classList.add("hidden");
        logoutBtn.classList.add("hidden");

        if (liveFeedRunning) stopLiveFeed();
    }

    async function handleAuthLogin(event) {
        if (event) event.preventDefault();
        authErrorMessage.textContent = "";

        const user = authUsername.value.trim();
        const pass = authPassword.value;

        if (!user || !pass) {
            authErrorMessage.textContent = "Please provide operator username and key.";
            return;
        }

        authSubmitBtn.disabled = true;
        authSubmitBtn.innerHTML = `<span class="banking-spinner" style="width:14px;height:14px;margin:0 8px 0 0;display:inline-block;vertical-align:middle;border-width:2px;"></span> Validating Clearance...`;

        try {
            const data = await window.FraudAPI.login(user, pass);
            setOperatorAuthenticated(user, data.user_role);
            authUsername.value = "";
            authPassword.value = "";
            showToast(`Operator Clearance Approved. Welcome, ${user}.`, "success");
        } catch (err) {
            authErrorMessage.textContent = err.message || "Invalid credentials.";
            showToast("Security Clearance Denied: Invalid credentials.", "error");
        } finally {
            authSubmitBtn.disabled = false;
            authSubmitBtn.innerHTML = `<span class="btn-sheen"></span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg> <span>Authenticate & Unlock Terminal</span>`;
        }
    }

    function handleLogout() {
        window.FraudAPI.logout();
        setOperatorLoggedOut();
        showToast("Session terminated. Terminal locked.", "success");
    }

    // ============================================================
    // Cockpit Telemetry & Metrics Loading
    // ============================================================
    async function loadCockpitData() {
        try {
            await Promise.all([
                loadSystemHealth(),
                loadAlertSummaryData(),
                loadAlertsForDistribution(),
            ]);

            if (lastSyncTime) lastSyncTime.textContent = new Date().toLocaleTimeString();
        } catch (e) {
            console.error("Cockpit refresh failure:", e);
        }
    }

    async function loadSystemHealth() {
        try {
            const health = await window.FraudAPI.getHealth();
            const dbConnected = health.database && health.database.status === "connected";

            if (apiStatusPill) {
                apiStatusPill.textContent = "ONLINE";
                apiStatusPill.className = "telemetry-val text-green";
            }

            if (dbStatusPill) {
                const dbType = (health.database?.database_type || "PostgreSQL").toUpperCase();
                dbStatusPill.textContent = dbConnected ? dbType : "FALLBACK";
                dbStatusPill.className = dbConnected ? "telemetry-val text-green" : "telemetry-val text-yellow";
            }

            if (cockpitDbState && health.database) {
                cockpitDbState.textContent = `CONNECTED (${health.database.database_name || "fraud_detection"})`;
            }

            if (cutoffStatusPill && health.calibrated_threshold !== undefined) {
                cutoffStatusPill.textContent = `T = ${Number(health.calibrated_threshold).toFixed(2)}`;
            }
        } catch (e) {
            if (apiStatusPill) {
                apiStatusPill.textContent = "OFFLINE";
                apiStatusPill.className = "telemetry-val text-danger";
            }
        }
    }

    async function loadAlertSummaryData() {
        try {
            const summary = await window.FraudAPI.getAlertSummary();
            const total = Number(summary.total_alerts || 0);
            const newCases = Number(summary.new_alerts || 0);
            const underReview = Number(summary.under_review || 0);
            const resolved = Number(summary.resolved || 0);

            if (statTotalAlerts) statTotalAlerts.textContent = total;
            if (statNewAlerts) statNewAlerts.textContent = newCases;
            if (statUnderReview) statUnderReview.textContent = underReview;
            if (statResolved) statResolved.textContent = resolved;
            if (activeAlertBadge) activeAlertBadge.textContent = total;
        } catch (e) {
            console.error("Error reading alert summary:", e);
        }
    }

    async function loadAlertsForDistribution() {
        try {
            const response = await window.FraudAPI.getAlerts();
            cachedAlerts = Array.isArray(response?.alerts) ? response.alerts : [];

            // Calculate Severity Distribution
            let high = 0, med = 0, low = 0;
            cachedAlerts.forEach((a) => {
                const lvl = String(a.risk_level || "").toLowerCase();
                if (lvl === "high") high++;
                else if (lvl === "medium") med++;
                else if (lvl === "low") low++;
            });

            if (barHighCount) barHighCount.textContent = `${high} cases`;
            if (barMedCount) barMedCount.textContent = `${med} cases`;
            if (barLowCount) barLowCount.textContent = `${low} cases`;

            const sum = high + med + low;
            const hPct = sum > 0 ? (high / sum) * 100 : 0;
            const mPct = sum > 0 ? (med / sum) * 100 : 0;
            const lPct = sum > 0 ? (low / sum) * 100 : 0;

            if (barHighFill) barHighFill.style.width = `${hPct}%`;
            if (barMedFill) barMedFill.style.width = `${mPct}%`;
            if (barLowFill) barLowFill.style.width = `${lPct}%`;

            // Populate Recent Interceptions in Cockpit
            renderCockpitMiniAlerts(cachedAlerts.slice(0, 6));

        } catch (e) {
            console.error("Error loading distribution alerts:", e);
        }
    }

    function renderCockpitMiniAlerts(alerts) {
        if (!cockpitMiniAlerts) return;
        if (alerts.length === 0) {
            cockpitMiniAlerts.innerHTML = `<div class="mini-alert-item">No recent alerts recorded.</div>`;
            return;
        }

        cockpitMiniAlerts.innerHTML = "";
        alerts.forEach((a) => {
            const item = document.createElement("div");
            item.className = "mini-alert-item danger";
            const prob = (Number(a.fraud_probability || 0) * 100).toFixed(1);
            const score = Number(a.risk_score || 0).toFixed(1);

            item.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;">
                    <span class="mini-alert-tx">${escapeHtml(a.transaction_id)}</span>
                    <small class="text-muted">${formatTimeOnly(a.created_at)}</small>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span class="mini-alert-prob font-mono">${prob}% Prob</span>
                    <span class="badge-sev high">Score: ${score}</span>
                </div>
            `;
            cockpitMiniAlerts.appendChild(item);
        });
    }

    // ============================================================
    // Payment Simulator: Scenario Auto-Filling
    // ============================================================
    function populateWirePreset(type) {
        const rand = Math.floor(1000 + Math.random() * 9000);
        const isoNow = new Date().toISOString().slice(0, 16);
        document.getElementById("wireTimestamp").value = isoNow;

        if (type === "attack") {
            document.getElementById("wireTxId").value = `TX-ATTACK-${rand}`;
            document.getElementById("wireSender").value = `US-JPMC-902144`;
            document.getElementById("wireReceiver").value = `KY-BCM-889102`;
            document.getElementById("wireAmount").value = "8450.00";
            document.getElementById("wireBalance").value = "9100.00";
            document.getElementById("wireType").value = "Transfer";
            document.getElementById("wireLocation").value = "Florida";
            document.getElementById("wireDeviceId").value = `DEV-MALICIOUS-${rand}`;
            document.getElementById("wireIp").value = "185.220.101.5";
            document.getElementById("toggleDevice").value = "0"; // Unrecognized
            document.getElementById("toggleThreat").value = "1"; // Threat Flagged
            document.getElementById("toggleGeoMatch").value = "0"; // Location Mismatch
            wireDeviation.value = "3.40";
            wireDeviationBadge.textContent = "3.40x Multiple";
            wireThreshold.value = "0.42";
            wireThresholdBadge.textContent = "0.42 (Optimal 98.5% Recall)";
            showToast("Preset Loaded: High-Value Account Takeover Attack", "error");
        } else if (type === "anomaly") {
            document.getElementById("wireTxId").value = `TX-ANOMALY-${rand}`;
            document.getElementById("wireSender").value = `US-BOFA-550921`;
            document.getElementById("wireReceiver").value = `UK-BARC-774019`;
            document.getElementById("wireAmount").value = "1420.00";
            document.getElementById("wireBalance").value = "4500.00";
            document.getElementById("wireType").value = "Withdrawal";
            document.getElementById("wireLocation").value = "Texas";
            document.getElementById("wireDeviceId").value = `DEV-FOREIGN-${rand}`;
            document.getElementById("wireIp").value = "77.91.76.247";
            document.getElementById("toggleDevice").value = "0"; // Unrecognized
            document.getElementById("toggleThreat").value = "0"; // Clean
            document.getElementById("toggleGeoMatch").value = "0"; // Mismatch
            wireDeviation.value = "1.25";
            wireDeviationBadge.textContent = "1.25x Multiple";
            wireThreshold.value = "0.42";
            wireThresholdBadge.textContent = "0.42 (Optimal 98.5% Recall)";
            showToast("Preset Loaded: Geo-Location Mismatch Anomaly", "warning");
        } else if (type === "clean") {
            document.getElementById("wireTxId").value = `TX-SETTLE-${rand}`;
            document.getElementById("wireSender").value = `US-WELLS-881023`;
            document.getElementById("wireReceiver").value = `US-CITI-339104`;
            document.getElementById("wireAmount").value = "4800.00";
            document.getElementById("wireBalance").value = "128500.00";
            document.getElementById("wireType").value = "Deposit";
            document.getElementById("wireLocation").value = "New York";
            document.getElementById("wireDeviceId").value = `DEV-CORP-SECURE`;
            document.getElementById("wireIp").value = "192.168.1.50";
            document.getElementById("toggleDevice").value = "1"; // Recognized
            document.getElementById("toggleThreat").value = "0"; // Clean
            document.getElementById("toggleGeoMatch").value = "1"; // Match
            wireDeviation.value = "0.05";
            wireDeviationBadge.textContent = "0.05x Multiple";
            wireThreshold.value = "0.42";
            wireThresholdBadge.textContent = "0.42 (Optimal 98.5% Recall)";
            showToast("Preset Loaded: Verified Corporate Vendor Wire", "success");
        }
    }

    // ============================================================
    // Payment Simulator: Execution & ML Scoring
    // ============================================================
    async function executeWireScoring(event) {
        if (event) event.preventDefault();

        btnExecuteScoring.disabled = true;
        btnExecuteScoring.innerHTML = `<span class="banking-spinner" style="width:14px;height:14px;margin:0 8px 0 0;display:inline-block;vertical-align:middle;border-width:2px;"></span> Intercepting & Scoring via ML Pipeline...`;

        const txId = document.getElementById("wireTxId").value.trim();
        const timestamp = document.getElementById("wireTimestamp").value || new Date().toISOString();
        const sender = document.getElementById("wireSender").value.trim();
        const receiver = document.getElementById("wireReceiver").value.trim();
        const amount = parseFloat(document.getElementById("wireAmount").value) || 0;
        const balance = parseFloat(document.getElementById("wireBalance").value) || 0;
        const type = document.getElementById("wireType").value;
        const location = document.getElementById("wireLocation").value;
        const deviceId = document.getElementById("wireDeviceId").value.trim() || "DEV-UNASSIGNED";
        const ip = document.getElementById("wireIp").value.trim() || "127.0.0.1";
        const devRecog = parseInt(document.getElementById("toggleDevice").value, 10);
        const threatFlag = parseInt(document.getElementById("toggleThreat").value, 10);
        const geoMatch = parseInt(document.getElementById("toggleGeoMatch").value, 10);
        const deviation = parseFloat(wireDeviation.value) || 0.0;
        const threshold = parseFloat(wireThreshold.value) || 0.42;

        const payload = {
            transaction_id: txId,
            timestamp: new Date(timestamp).toISOString(),
            sender_account_id: sender,
            receiver_account_id: receiver,
            amount: amount,
            type: type,
            device_id: deviceId,
            ip_address: ip,
            location: location,
            account_balance: balance,
            user_device_recognition: devRecog,
            known_threat_flag: threatFlag,
            login_location_match: geoMatch,
            spending_pattern_deviation: deviation,
        };

        try {
            const result = await window.FraudAPI.predictTransaction(payload, threshold);

            // Display Results in HUD
            hudEmptyState.classList.add("hidden");
            hudLiveScored.classList.remove("hidden");
            hudTimestamp.textContent = `SCORED AT ${new Date().toLocaleTimeString()}`;

            const riskScore = Number(result.risk_score || 0);
            const fraudProb = Number(result.fraud_probability || 0);
            const isSuspicious = String(result.prediction).toLowerCase() === "suspicious";

            // Gauge SVG Arc Offset (Circumference: 298.45)
            const circumference = 298.45;
            const targetOffset = circumference - (riskScore / 100) * circumference;
            hudGaugeFill.style.strokeDashoffset = targetOffset;
            hudRiskScore.textContent = riskScore.toFixed(1);

            if (isSuspicious) {
                hudGaugeFill.style.stroke = "var(--text-danger)";
                hudVerdictBanner.className = "hud-verdict-banner verdict-fraud";
                hudVerdictIcon.textContent = "🚨";
                hudVerdictText.textContent = "SUSPICIOUS WIRE — INTERCEPTED";
                hudProbVal.className = "hud-t-value font-mono text-danger";
                hudTierVal.className = "hud-t-value text-danger";
                hudDbSyncVal.textContent = "AUTO-STORED IN POSTGRESQL";
                hudDbSyncVal.className = "hud-t-value text-green";

                playBeep("alert");
                showToast(`🚨 WIRE INTERCEPTED! Fraud Probability: ${(fraudProb * 100).toFixed(1)}%`, "error");
            } else {
                hudGaugeFill.style.stroke = "var(--text-green)";
                hudVerdictBanner.className = "hud-verdict-banner verdict-clean";
                hudVerdictIcon.textContent = "✅";
                hudVerdictText.textContent = "LEGITIMATE WIRE — APPROVED";
                hudProbVal.className = "hud-t-value font-mono text-green";
                hudTierVal.className = "hud-t-value text-green";
                hudDbSyncVal.textContent = "CLEARED (NO CASE OPENED)";
                hudDbSyncVal.className = "hud-t-value text-muted";

                playBeep("clean");
                showToast(`✅ Wire Approved. Settled with low risk score (${riskScore.toFixed(1)})`, "success");
            }

            hudProbVal.textContent = `${(fraudProb * 100).toFixed(2)}%`;
            hudTierVal.textContent = result.risk_level.toUpperCase();
            hudCutoffVal.textContent = `T = ${Number(result.decision_threshold).toFixed(2)}`;

            // Populate XAI Bullet Reasons
            renderHudXaiSignals(payload, isSuspicious, riskScore);

            // Trigger 3D Satellite Global Interception Laser Strike
            if (window.SatelliteDefense && typeof window.SatelliteDefense.triggerInterception === "function") {
                window.SatelliteDefense.triggerInterception({
                    transaction_id: payload.transaction_id,
                    location: payload.location,
                    amount: payload.amount,
                    risk_score: riskScore,
                    fraud_probability: fraudProb,
                    prediction: result.prediction,
                    risk_level: result.risk_level,
                    is_fraud: isSuspicious,
                    threat_flag: payload.known_threat_flag,
                    device_id: payload.device_id,
                    ip_address: payload.ip_address
                });
            }

            // Background reload cockpit counters
            loadAlertSummaryData();

        } catch (error) {
            showToast(`Scoring Error: ${error.message}`, "error");
        } finally {
            btnExecuteScoring.disabled = false;
            btnExecuteScoring.innerHTML = `<span class="btn-icon">⚡</span> Intercept & Score Payment via Machine Learning`;
        }
    }

    // ============================================================
    // Explainable AI (XAI) Signals Generator
    // ============================================================
    function renderHudXaiSignals(tx, isSuspicious, score) {
        hudXaiList.innerHTML = "";
        const signals = [];

        if (tx.known_threat_flag === 1) {
            signals.push({
                badge: "CRITICAL",
                type: "crit",
                text: "Originating IP or account matched on active AML / international cybercrime blacklist (+40% risk)."
            });
        }

        if (tx.user_device_recognition === 0) {
            signals.push({
                badge: "HARDWARE",
                type: "crit",
                text: "Unrecognized hardware fingerprint: First-time terminal used for wire transfer (+28% risk)."
            });
        }

        if (tx.login_location_match === 0) {
            signals.push({
                badge: "GEO ANOMALY",
                type: "warn",
                text: `Login session originated from ${tx.location}, conflicting with verified home jurisdiction.`
            });
        }

        if (tx.account_balance > 0) {
            const drainRatio = (tx.amount / tx.account_balance) * 100;
            if (drainRatio >= 75) {
                signals.push({
                    badge: "CAPITAL DRAIN",
                    type: "crit",
                    text: `Rapid capital exhaustion: Wire of $${tx.amount.toLocaleString()} demands ${drainRatio.toFixed(1)}% of total account funds.`
                });
            }
        }

        if (tx.spending_pattern_deviation >= 2.0) {
            signals.push({
                badge: "VELOCITY SPIKE",
                type: "warn",
                text: `Behavioral velocity is ${tx.spending_pattern_deviation.toFixed(2)}x standard deviations above historical mean volume.`
            });
        } else if (tx.spending_pattern_deviation >= 1.0) {
            signals.push({
                badge: "VELOCITY",
                type: "warn",
                text: `Elevated transaction deviation: ${tx.spending_pattern_deviation.toFixed(2)}x above regular activity.`
            });
        }

        if (signals.length === 0) {
            signals.push({
                badge: "VERIFIED",
                type: "safe",
                text: "All credentials, hardware fingerprints, behavioral velocities, and geo-locations verified within legitimate tolerances."
            });
        }

        signals.forEach((s) => {
            const li = document.createElement("li");
            li.className = "xai-bullet";
            li.innerHTML = `
                <span class="xai-bullet-badge ${s.type}">${s.badge}</span>
                <span>${escapeHtml(s.text)}</span>
            `;
            hudXaiList.appendChild(li);
        });
    }

    // ============================================================
    // Live Wire Stream (Real-Time Background Auto-Scorer)
    // ============================================================
    function toggleLiveFeed() {
        if (liveFeedRunning) {
            stopLiveFeed();
        } else {
            startLiveFeed();
        }
    }

    function startLiveFeed() {
        liveFeedRunning = true;
        liveFeedIcon.textContent = "⏸";
        liveFeedText.textContent = "Pause Auto Ingestion";
        btnToggleLiveFeed.className = "btn btn-outline-danger";
        showToast("Live Wire Ingestion active: Streaming incoming interbank payments...", "success");

        // Clear empty row if present
        const emptyRow = streamTableBody.querySelector(".stream-empty-row");
        if (emptyRow) emptyRow.remove();

        // Ingest immediately, then every 2.8 seconds
        ingestStreamTransaction();
        liveFeedInterval = setInterval(ingestStreamTransaction, 2800);
    }

    function stopLiveFeed() {
        liveFeedRunning = false;
        clearInterval(liveFeedInterval);
        liveFeedInterval = null;
        liveFeedIcon.textContent = "▶";
        liveFeedText.textContent = "Start Auto Ingestion";
        btnToggleLiveFeed.className = "btn btn-primary";
        showToast("Live Wire Ingestion paused.", "warning");
    }

    async function ingestStreamTransaction() {
        const rand = Math.floor(1000 + Math.random() * 9000);
        const isFraudProfile = Math.random() < 0.35; // 35% chance to simulate a fraud attack

        const locations = ["California", "New York", "Texas", "Florida"];
        const loc = locations[Math.floor(Math.random() * locations.length)];

        let amount, balance, devRecog, threatFlag, geoMatch, deviation;

        if (isFraudProfile) {
            amount = Math.floor(3500 + Math.random() * 6000);
            balance = amount + Math.floor(200 + Math.random() * 800);
            devRecog = 0;
            threatFlag = Math.random() < 0.6 ? 1 : 0;
            geoMatch = 0;
            deviation = parseFloat((2.0 + Math.random() * 2.5).toFixed(2));
        } else {
            amount = Math.floor(50 + Math.random() * 800);
            balance = Math.floor(8000 + Math.random() * 25000);
            devRecog = 1;
            threatFlag = 0;
            geoMatch = 1;
            deviation = parseFloat((0.05 + Math.random() * 0.4).toFixed(2));
        }

        const payload = {
            transaction_id: `TX-WIRE-${rand}`,
            timestamp: new Date().toISOString(),
            sender_account_id: `US-DEBTOR-${rand}`,
            receiver_account_id: `BENEFICIARY-${Math.floor(1000 + Math.random() * 9000)}`,
            amount: amount,
            type: "Transfer",
            device_id: `DEV-STREAM-${rand}`,
            ip_address: `198.51.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
            location: loc,
            account_balance: balance,
            user_device_recognition: devRecog,
            known_threat_flag: threatFlag,
            login_location_match: geoMatch,
            spending_pattern_deviation: deviation,
        };

        try {
            const res = await window.FraudAPI.predictTransaction(payload, 0.42);
            liveStreamTotal++;

            const isSuspicious = String(res.prediction).toLowerCase() === "suspicious";
            if (isSuspicious) {
                liveStreamFrauds++;
                liveStreamSavedCapital += amount;
                playBeep("alert");
            } else {
                liveStreamClean++;
            }

            // Update Statistics Ribbon
            if (streamCount) streamCount.textContent = liveStreamTotal;
            if (streamFraudCount) streamFraudCount.textContent = liveStreamFrauds;
            if (streamCleanCount) streamCleanCount.textContent = liveStreamClean;
            if (streamSavedCapital) streamSavedCapital.textContent = `$${liveStreamSavedCapital.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;

            // Prepend Row to Stream Table
            const row = document.createElement("tr");
            const prob = (Number(res.fraud_probability || 0) * 100).toFixed(1);

            row.innerHTML = `
                <td><code>${new Date().toLocaleTimeString()}</code></td>
                <td><strong>${escapeHtml(res.transaction_id)}</strong></td>
                <td><small class="font-mono">${escapeHtml(payload.sender_account_id)} &rarr; ${escapeHtml(payload.receiver_account_id)}</small></td>
                <td><strong>$${amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong></td>
                <td>${escapeHtml(loc)}</td>
                <td>${threatFlag === 1 ? '<span class="badge-sev high">BLACKLIST</span>' : '<span class="text-muted">Clean</span>'}</td>
                <td><strong class="font-mono ${isSuspicious ? 'text-danger' : 'text-green'}">${prob}%</strong></td>
                <td>
                    <span class="status-pill-badge ${isSuspicious ? 'status-new' : 'status-resolved'}">
                        ${isSuspicious ? '🚨 INTERCEPTED' : '✅ APPROVED'}
                    </span>
                </td>
            `;

            streamTableBody.insertBefore(row, streamTableBody.firstChild);

            // Trigger 3D Satellite Global Interception Laser Strike
            if (window.SatelliteDefense && typeof window.SatelliteDefense.triggerInterception === "function") {
                window.SatelliteDefense.triggerInterception({
                    transaction_id: payload.transaction_id,
                    location: payload.location,
                    amount: payload.amount,
                    risk_score: res.risk_score !== undefined ? res.risk_score : (isSuspicious ? 98.4 : 8.2),
                    fraud_probability: res.fraud_probability,
                    prediction: res.prediction,
                    risk_level: res.risk_level || (isSuspicious ? "high" : "low"),
                    is_fraud: isSuspicious,
                    threat_flag: threatFlag,
                    device_id: payload.device_id,
                    ip_address: payload.ip_address
                });
            }

            // Cap stream table at 25 rows for memory stability
            if (streamTableBody.children.length > 25) {
                streamTableBody.removeChild(streamTableBody.lastChild);
            }

            // Refresh summary counters
            loadAlertSummaryData();

        } catch (err) {
            console.error("Stream ingestion error:", err);
        }
    }

    function clearLiveFeed() {
        streamTableBody.innerHTML = `
            <tr class="stream-empty-row">
                <td colspan="8">Stream cleared. Waiting for transactions...</td>
            </tr>
        `;
        liveStreamTotal = 0;
        liveStreamFrauds = 0;
        liveStreamClean = 0;
        liveStreamSavedCapital = 0;

        if (streamCount) streamCount.textContent = "0";
        if (streamFraudCount) streamFraudCount.textContent = "0";
        if (streamCleanCount) streamCleanCount.textContent = "0";
        if (streamSavedCapital) streamSavedCapital.textContent = "$0.00";
    }

    // ============================================================
    // Batch Liquidity Settlement: 197 Certified Fraud Batch
    // ============================================================
    async function handleExecute197Batch() {
        btnExecute197Batch.disabled = true;
        batchLoadingIndicator.classList.remove("hidden");
        batchResultsHub.classList.add("hidden");

        const t0 = performance.now();

        try {
            const data = await window.FraudAPI.getSample197Batch();
            const txs = data.transactions || [];

            if (txs.length === 0) {
                throw new Error("Unable to locate 197 batch transactions file.");
            }

            const res = await window.FraudAPI.predictBatch(txs, 0.42);
            const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

            batchLatencyBadge.textContent = `Completed in ${elapsed}s (Latency: ${(elapsed / txs.length * 1000).toFixed(1)}ms/tx)`;
            bmTotal.textContent = res.summary.total_transactions;
            bmSuspicious.textContent = res.summary.suspicious_count;
            bmNormal.textContent = res.summary.normal_count;
            bmRate.textContent = `${res.summary.fraud_rate_percentage.toFixed(1)}%`;
            bmAvgRisk.textContent = `${res.summary.average_risk_score.toFixed(1)} / 100`;

            // Populate Preview Table
            batchPreviewTableBody.innerHTML = "";
            (res.predictions || []).slice(0, 15).forEach((p) => {
                const tr = document.createElement("tr");
                const prob = (Number(p.fraud_probability || 0) * 100).toFixed(2);
                const sev = String(p.risk_level || "high").toLowerCase();

                tr.innerHTML = `
                    <td><strong>${escapeHtml(p.transaction_id)}</strong></td>
                    <td><span class="status-pill-badge status-new">${escapeHtml(p.prediction)}</span></td>
                    <td><span class="font-mono font-bold text-danger">${prob}%</span></td>
                    <td><strong class="font-mono">${Number(p.risk_score).toFixed(1)}</strong></td>
                    <td><span class="badge-sev ${sev}">${escapeHtml(p.risk_level)}</span></td>
                `;
                batchPreviewTableBody.appendChild(tr);
            });

            batchResultsHub.classList.remove("hidden");
            playBeep("alert");
            showToast(`Batch Settlement Complete: All ${res.summary.suspicious_count} frauds intercepted and synchronized to PostgreSQL!`, "success");

            // Refresh Cockpit & Case Desk
            await loadCockpitData();
            await loadTriageCases();

        } catch (error) {
            showToast(`Batch execution failed: ${error.message}`, "error");
        } finally {
            batchLoadingIndicator.classList.add("hidden");
            btnExecute197Batch.disabled = false;
        }
    }

    // Custom Batch File Upload
    function handleCustomBatchFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        batchCustomFileLabel.textContent = file.name;
        const reader = new FileReader();

        reader.onload = async function (e) {
            try {
                const json = JSON.parse(e.target.result);
                const txs = Array.isArray(json) ? json : json.transactions;

                if (!Array.isArray(txs) || txs.length === 0) {
                    throw new Error("Uploaded JSON must contain an array under 'transactions'.");
                }

                btnExecute197Batch.disabled = true;
                batchLoadingIndicator.classList.remove("hidden");
                batchResultsHub.classList.add("hidden");

                const t0 = performance.now();
                const res = await window.FraudAPI.predictBatch(txs, 0.42);
                const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

                batchLatencyBadge.textContent = `Completed in ${elapsed}s`;
                bmTotal.textContent = res.summary.total_transactions;
                bmSuspicious.textContent = res.summary.suspicious_count;
                bmNormal.textContent = res.summary.normal_count;
                bmRate.textContent = `${res.summary.fraud_rate_percentage.toFixed(1)}%`;
                bmAvgRisk.textContent = `${res.summary.average_risk_score.toFixed(1)} / 100`;

                batchPreviewTableBody.innerHTML = "";
                (res.predictions || []).slice(0, 15).forEach((p) => {
                    const tr = document.createElement("tr");
                    const prob = (Number(p.fraud_probability || 0) * 100).toFixed(2);
                    const sev = String(p.risk_level || "high").toLowerCase();
                    tr.innerHTML = `
                        <td><strong>${escapeHtml(p.transaction_id)}</strong></td>
                        <td><span class="status-pill-badge status-new">${escapeHtml(p.prediction)}</span></td>
                        <td><span class="font-mono font-bold text-danger">${prob}%</span></td>
                        <td><strong class="font-mono">${Number(p.risk_score).toFixed(1)}</strong></td>
                        <td><span class="badge-sev ${sev}">${escapeHtml(p.risk_level)}</span></td>
                    `;
                    batchPreviewTableBody.appendChild(tr);
                });

                batchResultsHub.classList.remove("hidden");
                showToast(`Custom clearing file processed: ${res.summary.suspicious_count} transactions intercepted.`, "success");

                await loadCockpitData();
                await loadTriageCases();

            } catch (err) {
                showToast(`File error: ${err.message}`, "error");
            } finally {
                batchLoadingIndicator.classList.add("hidden");
                btnExecute197Batch.disabled = false;
            }
        };

        reader.readAsText(file);
    }

    // ============================================================
    // Fraud Case Investigation (Triage Desk)
    // ============================================================
    async function loadTriageCases() {
        if (!triageTableBody) return;
        triageLoadingMsg.classList.remove("hidden");
        triageErrorMsg.textContent = "";

        try {
            const statusVal = triageStatusFilter ? triageStatusFilter.value : "";
            const response = await window.FraudAPI.getAlerts(statusVal);
            cachedAlerts = Array.isArray(response?.alerts) ? response.alerts : [];
            renderTriageTable();
        } catch (e) {
            triageErrorMsg.textContent = e.message || "Failed to load database cases.";
            triageTableBody.innerHTML = `<tr><td colspan="9" class="empty-state">Unable to communicate with PostgreSQL alerts database.</td></tr>`;
        } finally {
            triageLoadingMsg.classList.add("hidden");
        }
    }

    function renderTriageTable() {
        const riskVal = triageRiskFilter ? triageRiskFilter.value.toLowerCase() : "";
        const searchVal = triageSearchInput ? triageSearchInput.value.trim().toLowerCase() : "";

        const filtered = cachedAlerts.filter((a) => {
            const matchRisk = !riskVal || String(a.risk_level || "").toLowerCase() === riskVal;
            const matchSearch = !searchVal || String(a.transaction_id || "").toLowerCase().includes(searchVal);
            return matchRisk && matchSearch;
        });

        if (triageCountLabel) {
            triageCountLabel.textContent = `Displaying ${filtered.length} of ${cachedAlerts.length} recorded fraud cases`;
        }

        triageTableBody.innerHTML = "";

        if (filtered.length === 0) {
            triageTableBody.innerHTML = `<tr><td colspan="9" class="empty-state">No matching fraud cases found in PostgreSQL audit log.</td></tr>`;
            return;
        }

        filtered.forEach((a) => {
            const tr = document.createElement("tr");
            const prob = (Number(a.fraud_probability || 0) * 100).toFixed(2);
            const score = Number(a.risk_score || 0).toFixed(1);
            const sevClass = String(a.risk_level || "high").toLowerCase();
            const statusClass = String(a.status || "new").toLowerCase().replace(/\s+/g, "-");

            tr.innerHTML = `
                <td><code>#${escapeHtml(a.id)}</code></td>
                <td><strong class="font-mono">${escapeHtml(a.transaction_id)}</strong></td>
                <td><strong class="font-mono text-danger">${prob}%</strong></td>
                <td><span class="status-pill-badge status-new">${escapeHtml(a.prediction || "Suspicious")}</span></td>
                <td><strong class="font-mono">${score}</strong></td>
                <td><span class="badge-sev ${sevClass}">${escapeHtml(a.risk_level || "High")}</span></td>
                <td><span class="status-pill-badge status-${statusClass}">${escapeHtml(a.status || "New")}</span></td>
                <td><small>${formatDateTime(a.created_at)}</small></td>
                <td class="triage-actions">
                    <button type="button" class="btn-triage-act act-review" onclick="window.BankingTerminal.updateCaseStatus(${a.id}, 'Under Review')" title="Assign Case to Investigator">🔍 Review</button>
                    <button type="button" class="btn-triage-act act-resolve" onclick="window.BankingTerminal.updateCaseStatus(${a.id}, 'Resolved')" title="Authorize & Close Case">✅ Settle</button>
                    <button type="button" class="btn-triage-act act-del" onclick="window.BankingTerminal.deleteCase(${a.id})" title="Purge Record">🗑️</button>
                </td>
            `;

            triageTableBody.appendChild(tr);
        });
    }

    async function updateCaseStatus(id, newStatus) {
        try {
            await window.FraudAPI.updateAlertStatus(id, newStatus);
            showToast(`Case #${id} status changed to '${newStatus}'.`, "success");
            await loadAlertSummaryData();
            await loadTriageCases();
        } catch (e) {
            showToast(`Update failed: ${e.message}`, "error");
        }
    }

    async function deleteCase(id) {
        if (!confirm(`Confirm permanent deletion of Fraud Case #${id} from PostgreSQL?`)) return;
        try {
            await window.FraudAPI.deleteAlert(id);
            showToast(`Case #${id} removed from database.`, "success");
            await loadAlertSummaryData();
            await loadTriageCases();
        } catch (e) {
            showToast(`Deletion error: ${e.message}`, "error");
        }
    }

    function exportTriageCsv() {
        if (!cachedAlerts || cachedAlerts.length === 0) {
            showToast("No cases available for export.", "error");
            return;
        }

        const headers = ["Case_ID", "Transaction_Ref", "Fraud_Probability", "Prediction", "Risk_Score", "Severity_Level", "Status", "Timestamp_Created"];
        const rows = cachedAlerts.map(a => [
            a.id,
            `"${a.transaction_id}"`,
            a.fraud_probability,
            `"${a.prediction}"`,
            a.risk_score,
            `"${a.risk_level}"`,
            `"${a.status}"`,
            `"${a.created_at}"`
        ]);

        const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `apex_fraud_audit_log_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        showToast("Audit Log CSV successfully exported.", "success");
    }

    // ============================================================
    // Formatters & Escapers
    // ============================================================
    function formatDateTime(val) {
        if (!val) return "--";
        const d = new Date(val);
        if (Number.isNaN(d.getTime())) return String(val);
        return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function formatTimeOnly(val) {
        if (!val) return "--:--";
        const d = new Date(val);
        if (Number.isNaN(d.getTime())) return String(val);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function escapeHtml(val) {
        if (val === null || val === undefined) return "";
        return String(val)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // ============================================================
    // Register Listeners
    // ============================================================
    function registerTerminalListeners() {
        // Navigation Tabs
        document.querySelectorAll(".bank-nav-tab").forEach((tab) => {
            tab.addEventListener("click", () => {
                switchTab(tab.getAttribute("data-tab"));
            });
        });

        // Authentication
        if (bankingAuthForm) bankingAuthForm.addEventListener("submit", handleAuthLogin);
        if (logoutBtn) logoutBtn.addEventListener("click", handleLogout);
        if (systemRefreshBtn) systemRefreshBtn.addEventListener("click", () => {
            loadCockpitData();
            showToast("System telemetry synchronized with PostgreSQL.", "success");
        });

        // Sound Toggle
        if (soundToggleBtn) {
            soundToggleBtn.addEventListener("click", () => {
                soundEnabled = !soundEnabled;
                soundIcon.textContent = soundEnabled ? "🔊" : "🔇";
                showToast(`Audio alerts ${soundEnabled ? "enabled" : "muted"}.`, "success");
            });
        }

        // Quick Role Login Buttons
        if (btnAdminQuickLogin) {
            btnAdminQuickLogin.addEventListener("click", () => {
                authUsername.value = "admin";
                authPassword.value = "admin123";
                handleAuthLogin();
            });
        }
        if (btnAnalystQuickLogin) {
            btnAnalystQuickLogin.addEventListener("click", () => {
                authUsername.value = "analyst";
                authPassword.value = "analyst123";
                handleAuthLogin();
            });
        }

        // Presets
        if (btnPresetAttack) btnPresetAttack.addEventListener("click", () => populateWirePreset("attack"));
        if (btnPresetAnomaly) btnPresetAnomaly.addEventListener("click", () => populateWirePreset("anomaly"));
        if (btnPresetClean) btnPresetClean.addEventListener("click", () => populateWirePreset("clean"));

        // New Wire ID Generator
        if (btnGenNewWireId) {
            btnGenNewWireId.addEventListener("click", () => {
                const rand = Math.floor(1000 + Math.random() * 9000);
                document.getElementById("wireTxId").value = `TX-WIRE-${rand}`;
            });
        }

        // Sliders
        if (wireDeviation) {
            wireDeviation.addEventListener("input", (e) => {
                wireDeviationBadge.textContent = `${parseFloat(e.target.value).toFixed(2)}x Multiple`;
            });
        }
        if (wireThreshold) {
            wireThreshold.addEventListener("input", (e) => {
                wireThresholdBadge.textContent = `${parseFloat(e.target.value).toFixed(2)} Cutoff`;
            });
        }

        // Payment Scoring Form
        if (wireTransferForm) wireTransferForm.addEventListener("submit", executeWireScoring);

        // Live Feed Controls
        if (btnToggleLiveFeed) btnToggleLiveFeed.addEventListener("click", toggleLiveFeed);
        if (btnClearLiveFeed) btnClearLiveFeed.addEventListener("click", clearLiveFeed);

        // Batch Controls
        if (btnExecute197Batch) btnExecute197Batch.addEventListener("click", handleExecute197Batch);
        if (batchCustomFileInput) batchCustomFileInput.addEventListener("change", handleCustomBatchFile);

        // Triage Controls
        if (triageStatusFilter) triageStatusFilter.addEventListener("change", loadTriageCases);
        if (triageRiskFilter) triageRiskFilter.addEventListener("change", renderTriageTable);
        if (triageSearchInput) triageSearchInput.addEventListener("input", renderTriageTable);
        if (btnReloadTriage) btnReloadTriage.addEventListener("click", () => {
            loadTriageCases();
            showToast("Cases reloaded from PostgreSQL.", "success");
        });
        if (btnExportTriageCsv) btnExportTriageCsv.addEventListener("click", exportTriageCsv);

        // Guide Explainer Modal
        const btnOpenGuideModal = document.getElementById("btnOpenGuideModal");
        const btnCloseGuideModal = document.getElementById("btnCloseGuideModal");
        const guideExplainerModal = document.getElementById("guideExplainerModal");

        if (btnOpenGuideModal && guideExplainerModal) {
            btnOpenGuideModal.addEventListener("click", () => {
                guideExplainerModal.classList.remove("hidden");
            });
        }
        if (btnCloseGuideModal && guideExplainerModal) {
            btnCloseGuideModal.addEventListener("click", () => {
                guideExplainerModal.classList.add("hidden");
            });
        }

        // Color Theme Palette Switcher
        const themeSelector = document.getElementById("themeSelector");
        const savedTheme = localStorage.getItem("bankingTheme") || "gold";
        document.body.setAttribute("data-theme", savedTheme);
        if (themeSelector) {
            themeSelector.value = savedTheme;
            themeSelector.addEventListener("change", (e) => {
                const newTheme = e.target.value;
                document.body.setAttribute("data-theme", newTheme);
                localStorage.setItem("bankingTheme", newTheme);
                const themeNames = {
                    gold: "Executive Gold",
                    cyber: "Cyber Threat (Red/Neon)",
                    obsidian: "Platinum Obsidian",
                    light: "Swiss Institutional Light"
                };
                showToast(`Color Palette: ${themeNames[newTheme] || newTheme} activated`, "success");
            });
        }
    }

    // Export to Window for inline button bindings
    window.BankingTerminal = {
        switchTab,
        updateCaseStatus,
        deleteCase,
        populateWirePreset,
    };

    // Initialize on DOM Ready
    document.addEventListener("DOMContentLoaded", () => {
        initMotionCanvas();
        registerTerminalListeners();

        // Default timestamp input
        const ts = document.getElementById("wireTimestamp");
        if (ts) ts.value = new Date().toISOString().slice(0, 16);

        if (window.FraudAPI && window.FraudAPI.isAuthenticated()) {
            setOperatorAuthenticated("ADMIN", "CHIEF RISK OFFICER");
        } else {
            setOperatorLoggedOut();
        }
    });

})();