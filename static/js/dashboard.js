document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("tx-form");
    const resultBox = document.getElementById("result-box");
    const riskBadge = document.getElementById("risk-badge");
    const riskScore = document.getElementById("risk-score");
    const actionTaken = document.getElementById("action-taken");
    const auditFlags = document.getElementById("audit-flags");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const payload = {
            user_id: document.getElementById("user_id").value,
            amount: parseFloat(document.getElementById("amount").value),
            merchant_category: document.getElementById("merchant_category").value,
            location: document.getElementById("location").value,
            device_type: "Mobile",
            transaction_frequency_24h: 3,
            user_avg_amount: 150.00
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            resultBox.classList.remove("hidden");
            riskBadge.innerText = data.risk_level + " RISK";
            riskBadge.style.backgroundColor = data.risk_level === "HIGH" ? "#ef4444" : (data.risk_level === "MEDIUM" ? "#f59e0b" : "#22c55e");
            riskScore.innerText = `Risk Score: ${data.risk_score}%`;
            actionTaken.innerText = `Action Taken: ${data.action_taken}`;
            
            auditFlags.innerHTML = "";
            data.audit_flags.forEach(flag => {
                const li = document.createElement("li");
                li.innerText = flag;
                auditFlags.appendChild(li);
            });
        } catch (err) {
            console.error("Error analyzing transaction:", err);
        }
    });
});
