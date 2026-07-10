// Global variables
let riskChart = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    // Current date for medical report
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('report-date').innerText = today;
});

// Navigate back to the Inputs Page
function navigateToInputs() {
    document.getElementById("results-page").classList.add("hidden");
    document.getElementById("btn-print-report").style.display = "none";
    document.getElementById("input-page").classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Reset Dashboard to initial state
function resetDashboard() {
    document.getElementById("metrics-form").reset();
    navigateToInputs();
}

// Handle Form Submission and navigate to Page 2
function handleFormSubmit(event) {
    event.preventDefault();

    // Show loading overlay
    const loadingOverlay = document.getElementById("loading-overlay");
    loadingOverlay.classList.add("active");

    // Gather inputs
    const age = parseInt(document.getElementById("age").value);
    const gender = document.getElementById("gender").value;
    const height = parseFloat(document.getElementById("height").value);
    const weight = parseFloat(document.getElementById("weight").value);
    const sbp = parseInt(document.getElementById("ap-hi").value);
    const dbp = parseInt(document.getElementById("ap-low").value);
    const cholesterol = parseInt(document.getElementById("cholesterol").value);
    const glucose = parseInt(document.getElementById("glucose").value);
    const smoke = document.getElementById("smoke").checked;
    const alco = document.getElementById("alco").checked;
    const active = document.getElementById("active").checked;

    // Simulate analysis delay (1.2 seconds) to display scanning effect
    setTimeout(() => {
        // 1. BMI Calculation
        const bmi = weight / Math.pow(height / 100, 2);
        let bmiClass = "Normal";
        let bmiStatusClass = "normal";
        if (bmi < 18.5) {
            bmiClass = "Underweight";
            bmiStatusClass = "warning";
        } else if (bmi >= 18.5 && bmi < 25) {
            bmiClass = "Normal Weight";
            bmiStatusClass = "normal";
        } else if (bmi >= 25 && bmi < 30) {
            bmiClass = "Overweight";
            bmiStatusClass = "warning";
        } else {
            bmiClass = "Obese";
            bmiStatusClass = "danger";
        }

        // 2. Blood Pressure Classification
        let bpClass = "Normal";
        let bpStatusClass = "normal";
        if (sbp < 120 && dbp < 80) {
            bpClass = "Normal";
            bpStatusClass = "normal";
        } else if (sbp >= 120 && sbp < 130 && dbp < 80) {
            bpClass = "Elevated";
            bpStatusClass = "warning";
        } else if ((sbp >= 130 && sbp < 140) || (dbp >= 80 && dbp < 90)) {
            bpClass = "Stage 1 Hypertension";
            bpStatusClass = "warning";
        } else if ((sbp >= 140 && sbp <= 180) || (dbp >= 90 && dbp <= 120)) {
            bpClass = "Stage 2 Hypertension";
            bpStatusClass = "danger";
        } else if (sbp > 180 || dbp > 120) {
            bpClass = "Hypertensive Crisis";
            bpStatusClass = "danger";
        }

        // 3. Clinical Prediction Model (Logistic Regression weights)
        let z = -3.8;
        z += age * 0.042; // Age
        if (gender === "male") z += 0.38; // Gender
        
        // BMI risk
        if (bmi >= 25 && bmi < 30) z += 0.25;
        if (bmi >= 30) z += 0.6;

        // BP risk
        if (bpClass === "Elevated") z += 0.3;
        if (bpClass === "Stage 1 Hypertension") z += 0.75;
        if (bpClass === "Stage 2 Hypertension") z += 1.45;
        if (bpClass === "Hypertensive Crisis") z += 2.3;

        // Pulse Pressure risk
        const pulsePressure = sbp - dbp;
        if (pulsePressure > 50) {
            z += (pulsePressure - 50) * 0.015;
        }

        // Cholesterol risk
        if (cholesterol === 2) z += 0.55;
        if (cholesterol === 3) z += 1.2;

        // Glucose risk
        if (glucose === 2) z += 0.35;
        if (glucose === 3) z += 0.85;

        // Lifestyle risk
        if (smoke) z += 0.8;
        if (alco) z += 0.15;
        if (!active) z += 0.45;

        // Map log-odds to probability
        let riskProbability = 1 / (1 + Math.exp(-z));
        let riskPercent = Math.round(riskProbability * 100);
        
        if (riskPercent < 2) riskPercent = 2; 
        if (riskPercent > 99) riskPercent = 99;

        // Determine Risk Category
        let riskCategory = "Low Risk";
        let riskColorClass = "low";
        let riskColorHex = "#10b981";
        let riskTextDescription = "";

        if (riskPercent < 10) {
            riskCategory = "Low Risk";
            riskColorClass = "low";
            riskColorHex = "#10b981";
            riskTextDescription = "Your calculated risk indicates a low overall probability of developing cardiovascular disease in the next 10 years. Maintain healthy diet and active habits.";
        } else if (riskPercent >= 10 && riskPercent < 25) {
            riskCategory = "Moderate Risk";
            riskColorClass = "mod";
            riskColorHex = "#f59e0b";
            riskTextDescription = "Your calculated risk is moderate. Focus on actionable improvements in blood pressure, cholesterol levels, or physical activity to reduce cardiovascular load.";
        } else {
            riskCategory = "High Risk";
            riskColorClass = "high";
            riskColorHex = "#ef4444";
            riskTextDescription = "Your calculated risk profile is elevated. We recommend checking in with a healthcare provider or cardiologist to establish lipid controls and diagnostic tracking.";
        }

        // 4. Switch Page Views (Page 1 -> Page 2)
        loadingOverlay.classList.remove("active");
        document.getElementById("input-page").classList.add("hidden");
        document.getElementById("results-page").classList.remove("hidden");
        document.getElementById("btn-print-report").style.display = "inline-flex";
        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Update Page 2 Core Results
        document.getElementById("risk-percent").innerText = `${riskPercent}%`;
        document.getElementById("patient-summary").innerText = `${gender === 'male' ? 'Male' : 'Female'} Patient, ${age} Years`;
        document.getElementById("risk-description").innerText = riskTextDescription;

        // Risk Badge
        const indicator = document.getElementById("risk-indicator");
        indicator.className = `risk-badge ${riskColorClass}`;
        indicator.innerText = riskCategory;

        // Update SVG Radial Gauge
        const radialGauge = document.getElementById("risk-gauge");
        const radius = 90;
        const circumference = 2 * Math.PI * radius; // ~565.48
        const offset = circumference - (riskPercent / 100) * circumference;
        radialGauge.style.strokeDashoffset = offset;
        radialGauge.style.stroke = riskColorHex;

        // 5. Populate Vitals Summary Pills (Previous Inputs Details)
        const summaryPillBox = document.getElementById("summary-pill-box");
        summaryPillBox.innerHTML = ""; // Clear

        const cholMap = { 1: "Normal", 2: "Above Normal", 3: "Well Above Normal" };
        const glucMap = { 1: "Normal", 2: "Above Normal", 3: "Well Above Normal" };

        const pills = [
            `Age: <strong>${age} yrs</strong>`,
            `Gender: <strong>${gender === 'male' ? 'Male' : 'Female'}</strong>`,
            `Height: <strong>${height} cm</strong>`,
            `Weight: <strong>${weight} kg</strong>`,
            `Blood Pressure: <strong>${sbp}/${dbp} mmHg</strong>`,
            `Cholesterol: <strong>${cholMap[cholesterol]}</strong>`,
            `Glucose: <strong>${glucMap[glucose]}</strong>`,
            `Smoking: <strong>${smoke ? 'Yes' : 'No'}</strong>`,
            `Alcohol: <strong>${alco ? 'Yes' : 'No'}</strong>`,
            `Active: <strong>${active ? 'Yes' : 'No'}</strong>`
        ];

        pills.forEach(p => {
            const pillEl = document.createElement("div");
            pillEl.className = "summary-pill";
            pillEl.innerHTML = p;
            summaryPillBox.appendChild(pillEl);
        });

        // 6. Risk Drivers Analysis (What is causing the percentage to go high)
        const driversBox = document.getElementById("drivers-box");
        driversBox.innerHTML = ""; // Clear

        const drivers = [];

        // Check Blood Pressure driver
        if (sbp >= 140 || dbp >= 90) {
            drivers.push({
                name: "Hypertension (Stage 2)",
                status: "high",
                val: `${sbp}/${dbp} mmHg`,
                desc: "High arterial force severely strains cardiovascular walls."
            });
        } else if (sbp >= 130 || dbp >= 80) {
            drivers.push({
                name: "Elevated Blood Pressure",
                status: "mod",
                val: `${sbp}/${dbp} mmHg`,
                desc: "Borderline readings increase risk of chronic hypertension."
            });
        } else {
            drivers.push({
                name: "Healthy Blood Pressure",
                status: "low",
                val: `${sbp}/${dbp} mmHg`,
                desc: "Optimal hemodynamic force."
            });
        }

        // Check Cholesterol driver
        if (cholesterol === 3) {
            drivers.push({
                name: "High Cholesterol Level",
                status: "high",
                val: "Class 3",
                desc: "High lipids promote coronary plaque accumulation."
            });
        } else if (cholesterol === 2) {
            drivers.push({
                name: "Elevated Cholesterol",
                status: "mod",
                val: "Class 2",
                desc: "Mild risk of lipid buildup in arteries."
            });
        } else {
            drivers.push({
                name: "Optimal Cholesterol",
                status: "low",
                val: "Class 1",
                desc: "Healthy baseline lipid balance."
            });
        }

        // Check Glucose driver
        if (glucose === 3) {
            drivers.push({
                name: "High Glucose Level",
                status: "high",
                val: "Class 3",
                desc: "High blood sugar accelerates arterial damage."
            });
        } else if (glucose === 2) {
            drivers.push({
                name: "Elevated Glucose",
                status: "mod",
                val: "Class 2",
                desc: "Pre-diabetic glycemic profile ranges."
            });
        } else {
            drivers.push({
                name: "Optimal Glucose",
                status: "low",
                val: "Class 1",
                desc: "Healthy glucose clearance rate."
            });
        }

        // Check Weight / BMI driver
        if (bmi >= 30) {
            drivers.push({
                name: "Obese BMI Classification",
                status: "high",
                val: `${bmi.toFixed(1)} kg/m²`,
                desc: "Excess body mass strains heart output capacity."
            });
        } else if (bmi >= 25) {
            drivers.push({
                name: "Overweight BMI",
                status: "mod",
                val: `${bmi.toFixed(1)} kg/m²`,
                desc: "Slight excess body weight increases load."
            });
        } else if (bmi >= 18.5) {
            drivers.push({
                name: "Healthy BMI Range",
                status: "low",
                val: `${bmi.toFixed(1)} kg/m²`,
                desc: "Optimal height-to-weight balance."
            });
        }

        // Check Smoking driver
        if (smoke) {
            drivers.push({
                name: "Active Tobacco Use",
                status: "high",
                val: "Smoker",
                desc: "Nicotine causes immediate vessel restriction and plaque instability."
            });
        }

        // Check Activity driver
        if (!active) {
            drivers.push({
                name: "Sedentary Lifestyle",
                status: "mod",
                val: "Inactive",
                desc: "Inactivity reduces vascular flexibility and aerobic strength."
            });
        }

        // Check Age driver (Non-modifiable)
        if (age >= 60) {
            drivers.push({
                name: "Advanced Biological Age",
                status: "high",
                val: `${age} yrs`,
                desc: "Natural age-related arterial hardening (non-modifiable)."
            });
        } else if (age >= 45) {
            drivers.push({
                name: "Moderate Age Risk Factor",
                status: "mod",
                val: `${age} yrs`,
                desc: "Standard age-related cardiovascular risk progression."
            });
        }

        // Render Driver Cards
        drivers.forEach(d => {
            const card = document.createElement("div");
            card.className = `driver-card ${d.status}`;
            card.innerHTML = `
                <div class="driver-info">
                    <h5>${d.name}</h5>
                    <p>${d.desc}</p>
                </div>
                <div class="driver-value ${d.status}">${d.val}</div>
            `;
            driversBox.appendChild(card);
        });

        // 7. Dynamic Recommendations (How to reduce it)
        const recList = document.getElementById("recommendations-list");
        recList.innerHTML = ""; // Clear

        let recs = [];
        
        if (bpClass !== "Normal") {
            recs.push({
                type: "bp",
                icon: "❤️",
                title: "Reduce Blood Pressure",
                desc: `Sodium reduction (below 2000mg/day), dietary changes (DASH plan), and moderate physical activity can reduce systolic pressure by 5-10 mmHg. Speak with a doctor regarding clinical targets.`
            });
        }
        if (cholesterol > 1) {
            recs.push({
                type: "diet",
                icon: "🥗",
                title: "Reduce LDL Cholesterol",
                desc: `Introduce soluble fibers (oat bran, beans) and plant sterols. Minimize intake of saturated and trans-fats. Omega-3 fatty acid intake will help raise protective HDL.`
            });
        }
        if (glucose > 1) {
            recs.push({
                type: "diet",
                icon: "🩸",
                title: "Control Glucose Spikes",
                desc: "Switch to high-fiber complex grains (quinoa, brown rice) and eliminate refined sugars. Exercising shortly after meals improves glucose clearance."
            });
        }
        if (bmi >= 25) {
            recs.push({
                type: "lifestyle",
                icon: "⚖️",
                title: "Manage Body Weight",
                desc: `Aiming for a 5% to 10% weight reduction over 6 months significantly reduces arterial pressure and strain. Create a structured 300-500 kcal deficit.`
            });
        }
        if (smoke) {
            recs.push({
                type: "lifestyle",
                icon: "🚬",
                title: "Halt Nicotine Intake",
                desc: "Smoking cessation is the single most impactful action. Risk of myocardial infarction drops by 50% within just one year of quitting."
            });
        }
        if (!active) {
            recs.push({
                type: "lifestyle",
                icon: "🏃",
                title: "Increase Physical Training",
                desc: "Initiate cardiorespiratory conditioning. Target 30 minutes of brisk walking or cycling at least 5 days a week to lower resting heart rate and arterial stiffness."
            });
        }
        
        if (recs.length === 0) {
            recs.push({
                type: "diet",
                icon: "🌟",
                title: "Maintain Optimal Vitals",
                desc: "Continue supporting your cardiac health with regular exercise, a Mediterranean diet high in monounsaturated fats, and yearly diagnostic profiling."
            });
        }

        recs.forEach(r => {
            const item = document.createElement("div");
            item.className = `rec-item ${r.type}`;
            item.innerHTML = `
                <div class="rec-icon">${r.icon}</div>
                <div class="rec-content">
                    <h4>${r.title}</h4>
                    <p>${r.desc}</p>
                </div>
            `;
            recList.appendChild(item);
        });

        // 8. Populate printable PDF Clinical Report Data
        document.getElementById("report-age").innerText = age;
        document.getElementById("report-gender").innerText = gender;
        document.getElementById("report-bmi").innerText = bmi.toFixed(1);
        document.getElementById("report-bmi-class").innerText = bmiClass;
        
        let reportStatusText = `Patient is a ${age}-year-old ${gender}. Vitals indicate ${bpClass.toLowerCase()} (${sbp}/${dbp} mmHg), with cholesterol classified as ${cholMap[cholesterol].toLowerCase()} and glucose as ${glucMap[glucose].toLowerCase()}.`;
        document.getElementById("report-status-summary").innerText = reportStatusText;

        // Report Risk Banner
        const rBanner = document.getElementById("report-risk-banner");
        rBanner.className = `report-risk-banner ${riskColorClass}`;
        document.getElementById("report-risk-score").innerText = `${riskPercent}%`;
        document.getElementById("report-risk-label").innerText = `${riskCategory}`;
        
        let reportRiskDesc = `Calculated 10-year risk profile is ${riskPercent}%. `;
        if (riskCategory === "Low Risk") {
            reportRiskDesc += "General prognosis is favorable. Follow preventative habits and regular medical screenings.";
        } else if (riskCategory === "Moderate Risk") {
            reportRiskDesc += "Moderate risk indicators detected. Actionable behavioral changes (aerobic activity, sodium reductions, dietary improvements) are recommended to maintain arterial elasticity.";
        } else {
            reportRiskDesc += "Elevated risk parameters identified. Clinical consultation, lipid profiling, and therapeutic interventions may be indicated to mitigate potential coronary artery disease.";
        }
        document.getElementById("report-risk-desc").innerText = reportRiskDesc;

        // Table updates
        document.getElementById("report-val-sbp").innerText = sbp;
        document.getElementById("report-class-sbp").innerText = bpClass;
        document.getElementById("report-flag-sbp").innerText = sbp >= 130 ? "HIGH" : "Normal";
        document.getElementById("report-flag-sbp").className = sbp >= 140 ? "flag-danger" : (sbp >= 120 ? "flag-warning" : "flag-normal");

        document.getElementById("report-val-dbp").innerText = dbp;
        document.getElementById("report-class-dbp").innerText = bpClass;
        document.getElementById("report-flag-dbp").innerText = dbp >= 80 ? "HIGH" : "Normal";
        document.getElementById("report-flag-dbp").className = dbp >= 90 ? "flag-danger" : (dbp >= 80 ? "flag-warning" : "flag-normal");

        document.getElementById("report-val-chol").innerText = cholMap[cholesterol];
        document.getElementById("report-class-chol").innerText = `Class ${cholesterol}`;
        document.getElementById("report-flag-chol").innerText = cholesterol > 1 ? "ELEVATED" : "Normal";
        document.getElementById("report-flag-chol").className = cholesterol === 3 ? "flag-danger" : (cholesterol === 2 ? "flag-warning" : "flag-normal");

        document.getElementById("report-val-gluc").innerText = glucMap[glucose];
        document.getElementById("report-class-gluc").innerText = `Class ${glucose}`;
        document.getElementById("report-flag-gluc").innerText = glucose > 1 ? "ELEVATED" : "Normal";
        document.getElementById("report-flag-gluc").className = glucose === 3 ? "flag-danger" : (glucose === 2 ? "flag-warning" : "flag-normal");

        document.getElementById("report-val-bmi").innerText = bmi.toFixed(1);
        document.getElementById("report-class-bmi").innerText = bmiClass;
        document.getElementById("report-flag-bmi").innerText = bmi >= 25 ? "OVERWEIGHT" : (bmi < 18.5 ? "UNDERWEIGHT" : "Normal");
        document.getElementById("report-flag-bmi").className = bmi >= 30 ? "flag-danger" : (bmi >= 25 || bmi < 18.5 ? "flag-warning" : "flag-normal");

        document.getElementById("report-val-smoke").innerText = smoke ? "Smoker" : "Non-Smoker";
        document.getElementById("report-flag-smoke").innerText = smoke ? "WARNING" : "Normal";
        document.getElementById("report-flag-smoke").className = smoke ? "flag-danger" : "flag-normal";

        document.getElementById("report-val-active").innerText = active ? "Active" : "Sedentary";
        document.getElementById("report-flag-active").innerText = active ? "Normal" : "WARNING";
        document.getElementById("report-flag-active").className = active ? "flag-normal" : "flag-warning";

        // Narrative Advice Box
        let adviceNarrative = `Clinical Analysis Summary: The patient's risk is assessed at ${riskPercent}%. `;
        if (recs.length > 0) {
            adviceNarrative += "Primary clinical actions involve: " + recs.map(r => r.title.replace("Plan", "").replace("Guidelines", "").replace("Controls", "")).join(", ") + ". ";
        }
        adviceNarrative += "Advise periodic testing of serum glucose and lipid markers. Review lifestyle choices periodically.";
        document.getElementById("report-assessment-text").innerText = adviceNarrative;

        // 9. Render dynamic comparison chart
        renderChart(riskPercent);

    }, 1200);
}

// Render dynamic comparison chart in light theme
function renderChart(patientRisk) {
    const ctx = document.getElementById('riskChart').getContext('2d');
    
    if (riskChart) {
        riskChart.destroy();
    }

    // Chart.js Configuration for Light Theme
    riskChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Optimal Baseline', 'Patient Score', 'Uncontrolled Risk Profile'],
            datasets: [{
                label: 'Risk Percentage',
                data: [5, patientRisk, 75],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.25)', // Green
                    patientRisk < 10 ? 'rgba(16, 185, 129, 0.65)' : (patientRisk < 25 ? 'rgba(245, 158, 11, 0.65)' : 'rgba(239, 68, 68, 0.65)'),
                    'rgba(239, 68, 68, 0.2)'  // Red
                ],
                borderColor: [
                    '#10b981',
                    patientRisk < 10 ? '#10b981' : (patientRisk < 25 ? '#f59e0b' : '#ef4444'),
                    '#ef4444'
                ],
                borderWidth: 2,
                borderRadius: 8,
                barThickness: 45
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` ${context.parsed.y}% Probability`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#f1f5f9' // Light slate line
                    },
                    ticks: {
                        color: '#64748b', // Slate 500
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#475569', // Slate 600
                        font: {
                            family: 'Outfit',
                            weight: '500'
                        }
                    }
                }
            }
        }
    });
}
