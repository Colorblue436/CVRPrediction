// =====================================================
// Cormeum — app.js
// Two-page flow: Page 1 (form + preview) → Page 2 (full report)
// =====================================================

// Shared computed data
let _computed = null;

// ----- Init -----
document.addEventListener("DOMContentLoaded", () => {
    // Footer year
    const yr = document.getElementById("footer-year");
    if (yr) yr.textContent = `© ${new Date().getFullYear()} Cormeum. Educational tool — not medical advice.`;

    // Init segmented controls
    document.querySelectorAll(".segmented").forEach(seg => {
        const hiddenId = seg.id.replace("seg-", "");
        const hiddenInput = document.getElementById(hiddenId);
        seg.querySelectorAll(".seg-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                seg.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                if (hiddenInput) hiddenInput.value = btn.dataset.value;
            });
        });
    });
});

// ----- Navigation -----
function navigateToInputs() {
    document.getElementById("page-1").classList.remove("hidden");
    document.getElementById("page-2").classList.add("hidden");
    document.getElementById("nav-p1").classList.remove("hidden");
    document.getElementById("nav-p2").classList.add("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

function showPage2() {
    if (!_computed) return;
    document.getElementById("page-1").classList.add("hidden");
    document.getElementById("page-2").classList.remove("hidden");
    document.getElementById("nav-p1").classList.add("hidden");
    document.getElementById("nav-p2").classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
    renderPage2(_computed);
}

// ----- Reset -----
function resetForm() {
    document.getElementById("metrics-form").reset();
    _computed = null;

    document.querySelectorAll(".segmented").forEach(seg => {
        const btns = seg.querySelectorAll(".seg-btn");
        btns.forEach(b => b.classList.remove("active"));
        if (btns.length) btns[0].classList.add("active");
        const hid = document.getElementById(seg.id.replace("seg-", ""));
        if (hid && btns[0]) hid.value = btns[0].dataset.value;
    });
    // Restore gender default to Male
    const gSeg = document.getElementById("seg-gender");
    if (gSeg) {
        const btns = gSeg.querySelectorAll(".seg-btn");
        btns.forEach(b => b.classList.remove("active"));
        btns[1].classList.add("active");
        document.getElementById("gender").value = "male";
    }
    document.getElementById("empty-state").classList.remove("hidden");
    document.getElementById("result-preview").classList.add("hidden");
}

// ----- Form Submit -----
function handleFormSubmit(event) {
    event.preventDefault();
    const overlay = document.getElementById("loading-overlay");
    overlay.classList.add("active");

    // Required
    const age    = parseInt(document.getElementById("age").value);
    const gender = document.getElementById("gender").value;
    const height = parseFloat(document.getElementById("height").value);
    const weight = parseFloat(document.getElementById("weight").value);

    // Optional with defaults
    const sbpRaw = document.getElementById("ap-hi").value;
    const dbpRaw = document.getElementById("ap-low").value;
    const sbpDefaulted = sbpRaw === "";
    const dbpDefaulted = dbpRaw === "";
    const sbp = sbpDefaulted ? 120 : parseInt(sbpRaw);
    const dbp = dbpDefaulted ? 80  : parseInt(dbpRaw);

    const cholRaw = document.getElementById("cholesterol").value;
    const glucRaw = document.getElementById("glucose").value;
    const cholDefaulted = !cholRaw;
    const glucDefaulted = !glucRaw;
    const cholesterol = cholRaw ? parseInt(cholRaw) : 1;
    const glucose     = glucRaw ? parseInt(glucRaw) : 1;

    const smoke = document.getElementById("smoke").value === "true";

    setTimeout(() => {
        // 1. BMI
        const bmi = weight / Math.pow(height / 100, 2);
        let bmiClass = "Normal";
        if      (bmi < 18.5) bmiClass = "Underweight";
        else if (bmi < 25)   bmiClass = "Normal";
        else if (bmi < 30)   bmiClass = "Overweight";
        else                  bmiClass = "Obese";

        // 2. Blood Pressure Classification
        let bpClass = "Normal";
        if      (sbp < 120 && dbp < 80)                                   bpClass = "Normal";
        else if (sbp >= 120 && sbp < 130 && dbp < 80)                     bpClass = "Elevated";
        else if ((sbp >= 130 && sbp < 140) || (dbp >= 80 && dbp < 90))    bpClass = "Stage 1 Hypertension";
        else if ((sbp >= 140 && sbp <= 180) || (dbp >= 90 && dbp <= 120)) bpClass = "Stage 2 Hypertension";
        else if (sbp > 180 || dbp > 120)                                   bpClass = "Hypertensive Crisis";

        // 3. Logistic Risk Model
        let z = -3.8;
        z += age * 0.042;
        if (gender === "male") z += 0.38;
        if (bmi >= 25 && bmi < 30) z += 0.25;
        if (bmi >= 30) z += 0.6;
        if (bpClass === "Elevated")             z += 0.3;
        if (bpClass === "Stage 1 Hypertension") z += 0.75;
        if (bpClass === "Stage 2 Hypertension") z += 1.45;
        if (bpClass === "Hypertensive Crisis")  z += 2.3;
        const pp = sbp - dbp;
        if (pp > 50) z += (pp - 50) * 0.015;
        if (cholesterol === 2) z += 0.55;
        if (cholesterol === 3) z += 1.2;
        if (glucose === 2) z += 0.35;
        if (glucose === 3) z += 0.85;
        if (smoke) z += 0.8;

        let riskScore = Math.round((1 / (1 + Math.exp(-z))) * 100);
        riskScore = Math.max(2, Math.min(99, riskScore));

        // 4. Risk band
        let band = "low", bandColor = "var(--risk-low)", badgeLabel = "Low risk";
        if (riskScore >= 10 && riskScore < 25) {
            band = "mid";  bandColor = "var(--risk-mid)";  badgeLabel = "Moderate risk";
        } else if (riskScore >= 25) {
            band = "high"; bandColor = "var(--risk-high)"; badgeLabel = "Elevated risk";
        }

        // Store computed results for Page 2
        _computed = {
            age, gender, height, weight, sbp, dbp, bmi, bmiClass, bpClass,
            cholesterol, glucose, smoke, riskScore, band, bandColor, badgeLabel,
            sbpDefaulted, dbpDefaulted, cholDefaulted, glucDefaulted
        };

        // 5. Update Page 1 preview panel
        overlay.classList.remove("active");
        document.getElementById("empty-state").classList.add("hidden");
        document.getElementById("result-preview").classList.remove("hidden");

        const scoreEl = document.getElementById("score-number-preview");
        scoreEl.textContent = riskScore;
        scoreEl.style.color = bandColor;

        const badge = document.getElementById("risk-badge-preview");
        badge.textContent = badgeLabel;
        badge.className = `risk-badge ${band}`;

        const fill = document.getElementById("score-bar-preview");
        fill.style.width = "0%";
        requestAnimationFrame(() => requestAnimationFrame(() => {
            fill.style.width = riskScore + "%";
            fill.style.backgroundColor = bandColor;
        }));

        const cholMap = { 1: "Normal", 2: "Above normal", 3: "Well above normal" };
        document.getElementById("bmi-note-preview").innerHTML =
            `BMI <strong>${bmi.toFixed(1)}</strong> (${bmiClass}). Score reflects cholesterol (${cholMap[cholesterol]}), glucose (${cholMap[glucose]}), and lifestyle.`;

        // Scroll to result panel on mobile
        if (window.innerWidth < 1024) {
            document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "start" });
        }

    }, 800);
}

// ----- Render Page 2 -----
function renderPage2(d) {
    const { age, gender, height, weight, sbp, dbp, bmi, bmiClass, bpClass,
            cholesterol, glucose, smoke, riskScore, band, bandColor, badgeLabel,
            sbpDefaulted, dbpDefaulted, cholDefaulted, glucDefaulted } = d;

    const cholMap = { 1: "Normal", 2: "Above normal", 3: "Well above normal" };

    // Score card
    const scoreEl = document.getElementById("score-number-p2");
    scoreEl.textContent = riskScore;
    scoreEl.style.color = bandColor;

    const badge = document.getElementById("risk-badge-p2");
    badge.textContent = badgeLabel;
    badge.className = `risk-badge ${band}`;

    const fill = document.getElementById("score-bar-p2");
    fill.style.width = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => {
        fill.style.width = riskScore + "%";
        fill.style.backgroundColor = bandColor;
    }));

    document.getElementById("bmi-note-p2").innerHTML =
        `BMI <strong>${bmi.toFixed(1)}</strong> (${bmiClass}). Score reflects the combined weight of all inputs.`;

    // Gauge
    const arc = document.getElementById("gauge-arc");
    const circumference = 2 * Math.PI * 66; // r=66 → ~414.69
    const offset = circumference - (riskScore / 100) * circumference;
    requestAnimationFrame(() => requestAnimationFrame(() => {
        arc.style.strokeDashoffset = offset;
        arc.style.stroke = bandColor;
    }));
    const gaugePct = document.getElementById("gauge-pct");
    gaugePct.textContent = riskScore + "%";
    gaugePct.style.color = bandColor;

    // Vitals pills
    const pillGrid = document.getElementById("vitals-pill-grid");
    pillGrid.innerHTML = "";
    const pills = [
        { label: "Age", value: `${age} yrs`, def: false },
        { label: "Gender", value: gender === "male" ? "Male" : "Female", def: false },
        { label: "Height", value: `${height} cm`, def: false },
        { label: "Weight", value: `${weight} kg`, def: false },
        { label: "Systolic BP", value: `${sbp} mmHg`, def: sbpDefaulted },
        { label: "Diastolic BP", value: `${dbp} mmHg`, def: dbpDefaulted },
        { label: "Cholesterol", value: cholMap[cholesterol], def: cholDefaulted },
        { label: "Glucose", value: cholMap[glucose], def: glucDefaulted },
        { label: "Smoking", value: smoke ? "Smoker" : "Non-smoker", def: false },
    ];
    pills.forEach(p => {
        const el = document.createElement("div");
        el.className = "vitals-pill" + (p.def ? " defaulted" : "");
        el.innerHTML = `${p.label}: <strong>${p.value}</strong>${p.def ? " <em style='font-size:0.65rem'>(est.)</em>" : ""}`;
        pillGrid.appendChild(el);
    });

    // Risk Drivers
    renderDrivers(d);

    // Recommendations
    renderRecs(d);

    // Contributing Factors
    renderFactors(d, "factors-list-p2");
}

// ----- Risk Drivers -----
function renderDrivers({ age, gender, bmi, bpClass, sbp, dbp, cholesterol, glucose, smoke }) {
    const list = document.getElementById("drivers-list");
    list.innerHTML = "";

    const drivers = [];

    // Blood pressure
    if      (sbp >= 140 || dbp >= 90)  drivers.push({ name: "Hypertension (Stage 2+)", status: "high", val: `${sbp}/${dbp} mmHg`, desc: "High arterial force severely strains cardiovascular walls." });
    else if (sbp >= 130 || dbp >= 80)  drivers.push({ name: "Elevated Blood Pressure",   status: "mod",  val: `${sbp}/${dbp} mmHg`, desc: "Borderline readings increase risk of chronic hypertension." });
    else                               drivers.push({ name: "Healthy Blood Pressure",    status: "low",  val: `${sbp}/${dbp} mmHg`, desc: "Optimal hemodynamic force." });

    // Cholesterol
    if      (cholesterol === 3) drivers.push({ name: "High Cholesterol",     status: "high", val: "Well above normal", desc: "High lipids promote coronary plaque accumulation." });
    else if (cholesterol === 2) drivers.push({ name: "Elevated Cholesterol", status: "mod",  val: "Above normal",      desc: "Mild risk of lipid buildup in arteries." });
    else                        drivers.push({ name: "Optimal Cholesterol",  status: "low",  val: "Normal",            desc: "Healthy baseline lipid balance." });

    // Glucose
    if      (glucose === 3) drivers.push({ name: "High Glucose",     status: "high", val: "Well above normal", desc: "High blood sugar accelerates arterial damage." });
    else if (glucose === 2) drivers.push({ name: "Elevated Glucose", status: "mod",  val: "Above normal",      desc: "Pre-diabetic glycemic profile range." });
    else                    drivers.push({ name: "Optimal Glucose",  status: "low",  val: "Normal",            desc: "Healthy glucose clearance rate." });

    // BMI
    if      (bmi >= 30)   drivers.push({ name: "Obese BMI",     status: "high", val: `${bmi.toFixed(1)} kg/m²`, desc: "Excess body mass strains heart output capacity." });
    else if (bmi >= 25)   drivers.push({ name: "Overweight BMI",status: "mod",  val: `${bmi.toFixed(1)} kg/m²`, desc: "Slight excess body weight increases cardiac load." });
    else if (bmi >= 18.5) drivers.push({ name: "Healthy BMI",   status: "low",  val: `${bmi.toFixed(1)} kg/m²`, desc: "Optimal height-to-weight balance." });

    // Smoking
    if (smoke) drivers.push({ name: "Active Tobacco Use", status: "high", val: "Smoker", desc: "Nicotine causes immediate vessel restriction and plaque instability." });

    // Age
    if      (age >= 60) drivers.push({ name: "Advanced Age", status: "high", val: `${age} yrs`, desc: "Natural age-related arterial hardening." });
    else if (age >= 45) drivers.push({ name: "Moderate Age Risk", status: "mod", val: `${age} yrs`, desc: "Standard age-related cardiovascular risk progression." });

    drivers.forEach(dr => {
        const li = document.createElement("div");
        li.className = `driver-item ${dr.status}`;
        li.innerHTML = `
            <div class="driver-info">
                <h5>${dr.name}</h5>
                <p>${dr.desc}</p>
            </div>
            <span class="driver-val ${dr.status}">${dr.val}</span>
        `;
        list.appendChild(li);
    });
}

// ----- Recommendations -----
function renderRecs({ bpClass, cholesterol, glucose, bmi, smoke }) {
    const list = document.getElementById("rec-list");
    list.innerHTML = "";

    const recs = [];

    if (bpClass !== "Normal")  recs.push({ icon: "❤️", title: "Reduce Blood Pressure",   desc: "Reduce sodium intake (under 2g/day), follow a DASH diet, and aim for 30 min of moderate activity daily to lower systolic pressure by 5–10 mmHg." });
    if (cholesterol > 1)        recs.push({ icon: "🥗", title: "Lower LDL Cholesterol",    desc: "Add soluble fibre (oat bran, legumes) and plant sterols to your diet. Minimise saturated and trans fats. Omega-3s help raise protective HDL." });
    if (glucose > 1)            recs.push({ icon: "🩸", title: "Control Blood Glucose",    desc: "Switch to high-fibre complex carbohydrates (quinoa, lentils) and avoid refined sugars. Light exercise after meals significantly improves glucose clearance." });
    if (bmi >= 25)              recs.push({ icon: "⚖️", title: "Manage Body Weight",       desc: "A 5–10% weight reduction over 6 months measurably reduces arterial pressure and inflammation. Aim for a sustainable 300–500 kcal daily deficit." });
    if (smoke)                  recs.push({ icon: "🚬", title: "Stop Smoking",              desc: "Cessation is the single most impactful action. Myocardial infarction risk drops by 50% within one year of quitting." });
    if (recs.length === 0)      recs.push({ icon: "🌟", title: "Maintain Healthy Habits",  desc: "Your vitals look great! Keep supporting heart health with regular exercise, a Mediterranean diet, and yearly preventive check-ups." });

    recs.forEach(r => {
        const el = document.createElement("div");
        el.className = "rec-item";
        el.innerHTML = `
            <span class="rec-icon">${r.icon}</span>
            <div class="rec-content">
                <h4>${r.title}</h4>
                <p>${r.desc}</p>
            </div>
        `;
        list.appendChild(el);
    });
}

// ----- Contributing Factors -----
function renderFactors({ age, gender, bmi, bpClass, sbp, dbp, cholesterol, glucose, smoke, bandColor }, listId) {
    const list = document.getElementById(listId);
    list.innerHTML = "";

    const bpPctMap = { "Normal": 8, "Elevated": 28, "Stage 1 Hypertension": 52, "Stage 2 Hypertension": 82, "Hypertensive Crisis": 96 };
    const factors = [
        { name: "Blood pressure", pct: bpPctMap[bpClass] ?? 8, risk: (bpPctMap[bpClass] ?? 8) > 20, note: `${sbp}/${dbp} mmHg` },
        { name: "Age",            pct: Math.round(Math.min(Math.max((age - 18) / 62 * 100, 5), 95)), risk: age >= 45, note: `${age} yrs` },
        { name: "BMI",            pct: bmi >= 30 ? 70 : bmi >= 25 ? 38 : bmi < 18.5 ? 22 : 10, risk: bmi >= 25, note: bmi.toFixed(1) },
        { name: "Cholesterol",    pct: cholesterol === 3 ? 80 : cholesterol === 2 ? 44 : 9, risk: cholesterol > 1, note: ["","Normal","Above","Well above"][cholesterol] },
        { name: "Glucose",        pct: glucose === 3 ? 64 : glucose === 2 ? 33 : 8, risk: glucose > 1, note: ["","Normal","Above","Well above"][glucose] },
        { name: "Smoking",        pct: smoke ? 74 : 5, risk: smoke, note: smoke ? "Active smoker" : "Non-smoker" },
        { name: "Sex",            pct: gender === "male" ? 38 : 18, risk: gender === "male", note: gender === "male" ? "Male" : "Female" },
    ];

    factors.forEach(f => {
        const barColor = f.risk ? bandColor : "var(--risk-low)";
        const li = document.createElement("li");
        li.className = "factor-item";
        li.innerHTML = `
            <span class="factor-name">${f.name}</span>
            <div class="factor-bar-track">
                <div class="factor-bar-fill" style="width:0%; background-color:${barColor};" data-pct="${f.pct}"></div>
            </div>
            <span class="factor-note">${f.note}</span>
        `;
        list.appendChild(li);
    });

    requestAnimationFrame(() => requestAnimationFrame(() => {
        list.querySelectorAll(".factor-bar-fill").forEach(b => { b.style.width = b.dataset.pct + "%"; });
    }));
}
