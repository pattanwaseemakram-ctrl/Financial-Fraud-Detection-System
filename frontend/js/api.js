// ============================================================
// Financial Fraud Detection System
// API Communication Layer
// ============================================================

const API_BASE_URL = (typeof window !== "undefined" && window.location.protocol.startsWith("http") && (window.location.port === "8000" || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"))
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://127.0.0.1:8000";


// ============================================================
// Token Management
// ============================================================

function getAccessToken() {
    return localStorage.getItem("fraud_access_token");
}


function setAccessToken(token) {
    localStorage.setItem("fraud_access_token", token);
}


function clearAccessToken() {
    localStorage.removeItem("fraud_access_token");
}


function isAuthenticated() {
    return Boolean(getAccessToken());
}


// ============================================================
// Common Request Helper
// ============================================================

async function apiRequest(
    endpoint,
    options = {}
) {
    const token = getAccessToken();

    const headers = {
        ...(options.headers || {})
    };

    if (!headers["Content-Type"] && options.body) {
        headers["Content-Type"] = "application/json";
    }

    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch(
        `${API_BASE_URL}${endpoint}`,
        {
            ...options,
            headers
        }
    );

    let data = null;

    const contentType =
        response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        data = await response.json();
    } else {
        data = await response.text();
    }

    if (!response.ok) {

        const errorMessage =
            typeof data === "object" && data?.detail
                ? data.detail
                : `Request failed with status ${response.status}`;

        throw new Error(errorMessage);
    }

    return data;
}


// ============================================================
// Authentication
// ============================================================

async function login(username, password) {

    const formData = new URLSearchParams();

    formData.append("username", username);
    formData.append("password", password);

    const response = await fetch(
        `${API_BASE_URL}/auth/token`,
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/x-www-form-urlencoded"
            },

            body: formData.toString()
        }
    );

    const data = await response.json();

    if (!response.ok) {

        const errorMessage =
            data?.detail ||
            "Login failed. Please check your credentials.";

        throw new Error(errorMessage);
    }

    if (!data.access_token) {
        throw new Error(
            "Authentication succeeded but no access token was returned."
        );
    }

    setAccessToken(data.access_token);

    return data;
}


function logout() {
    clearAccessToken();
}


// ============================================================
// Alerts API
// ============================================================

async function getAlerts(status = "") {

    let endpoint = "/alerts";

    if (status) {
        endpoint += `?status=${encodeURIComponent(status)}`;
    }

    return await apiRequest(endpoint, {
        method: "GET"
    });
}


// ============================================================
// Alert Summary API
// ============================================================

async function getAlertSummary() {

    return await apiRequest(
        "/alerts-summary",
        {
            method: "GET"
        }
    );
}


// ============================================================
// Single Alert API
// ============================================================

async function getAlert(alertId) {

    return await apiRequest(
        `/alerts/${alertId}`,
        {
            method: "GET"
        }
    );
}


// ============================================================
// Update Alert Status
// ============================================================

async function updateAlertStatus(
    alertId,
    status
) {

    return await apiRequest(
        `/alerts/${alertId}/status`,
        {
            method: "PUT",

            body: JSON.stringify({
                status: status
            })
        }
    );
}


// ============================================================
// Delete Alert
// ============================================================

async function deleteAlert(alertId) {

    return await apiRequest(
        `/alerts/${alertId}`,
        {
            method: "DELETE"
        }
    );
}


// ============================================================
// Health Check
// ============================================================

async function getHealth() {

    return await apiRequest(
        "/health",
        {
            method: "GET"
        }
    );
}


// ============================================================
// Root API Status
// ============================================================

async function getApiStatus() {

    return await apiRequest(
        "/",
        {
            method: "GET"
        }
    );
}


// ============================================================
// Fraud Prediction
// ============================================================

async function predictTransaction(transaction, threshold = null) {
    let endpoint = "/predict";
    if (threshold !== null && threshold !== undefined && threshold !== "") {
        endpoint += `?threshold=${encodeURIComponent(threshold)}`;
    }

    return await apiRequest(
        endpoint,
        {
            method: "POST",
            body: JSON.stringify(transaction)
        }
    );
}


// ============================================================
// Batch Prediction
// ============================================================

async function predictBatch(transactions, threshold = null) {
    let endpoint = "/batch-predict";
    if (threshold !== null && threshold !== undefined && threshold !== "") {
        endpoint += `?threshold=${encodeURIComponent(threshold)}`;
    }

    return await apiRequest(
        endpoint,
        {
            method: "POST",
            body: JSON.stringify({
                transactions: transactions
            })
        }
    );
}


async function getSample197Batch() {
    return await apiRequest("/sample-197-batch", {
        method: "GET"
    });
}


// ============================================================
// Export Functions
// ============================================================

window.FraudAPI = {
    getAccessToken,
    setAccessToken,
    clearAccessToken,
    isAuthenticated,

    login,
    logout,

    getAlerts,
    getAlertSummary,
    getAlert,

    updateAlertStatus,
    deleteAlert,

    getHealth,
    getApiStatus,

    predictTransaction,
    predictBatch,
    getSample197Batch,
};