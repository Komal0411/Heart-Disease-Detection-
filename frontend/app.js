/* Heart Disease Risk Predictor — frontend logic (8-field model) */

const API = {
  metadata: "/api/metadata",
  predict: "/api/predict",
};

const EXAMPLES = {
  low: {
    age: 45, sex: "Female", cp: "non-anginal",
    trestbps: 120, chol: 180, thalch: 170,
    exang: false, oldpeak: 0.0,
  },
  high: {
    age: 63, sex: "Male", cp: "asymptomatic",
    trestbps: 160, chol: 290, thalch: 105,
    exang: true, oldpeak: 2.8,
  },
};

const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const emptyState = document.getElementById("emptyState");
const resultBody = document.getElementById("resultBody");

/* ── populate dropdowns from the API's metadata ───────────────────── */
function fillSelect(id, options, preferred) {
  const el = document.getElementById(id);
  el.innerHTML = "";
  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    el.appendChild(o);
  });
  if (preferred && options.includes(preferred)) el.value = preferred;
}

async function init() {
  try {
    const res = await fetch(API.metadata);
    if (!res.ok) throw new Error("metadata request failed");
    const meta = await res.json();
    const cats = meta.categories || {};

    fillSelect("sex", cats.sex || ["Female", "Male"], "Male");
    fillSelect("cp", cats.cp || [], "asymptomatic");

    const m = meta.selected_metrics || {};
    document.getElementById("modelBadge").textContent =
      `${meta.model} · AUC ${(m.roc_auc ?? 0).toFixed(3)} · ${meta.n_samples} records`;
    document.getElementById("footerModel").textContent =
      `Model: ${meta.model} · Dataset: ${meta.dataset} · ${meta.n_features} features`;
  } catch (err) {
    document.getElementById("modelBadge").textContent = "model unavailable";
    console.error(err);
  }
}

/* ── toggle label ────────────────────────────────────────────────── */
const exangBox = document.getElementById("exang");
const exangText = document.getElementById("exangText");
exangBox.addEventListener("change", () => {
  exangText.textContent = exangBox.checked ? "Yes" : "No";
});

/* ── examples ────────────────────────────────────────────────────── */
document.querySelectorAll("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const data = EXAMPLES[btn.dataset.example];
    Object.entries(data).forEach(([key, value]) => {
      const field = form.elements[key];
      if (!field) return;
      if (field.type === "checkbox") {
        field.checked = Boolean(value);
        field.dispatchEvent(new Event("change"));
      } else {
        field.value = value;
      }
    });
  });
});

/* ── submit ──────────────────────────────────────────────────────── */
form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    age: Number(form.elements["age"].value),
    sex: form.elements["sex"].value,
    cp: form.elements["cp"].value,
    trestbps: Number(form.elements["trestbps"].value),
    chol: Number(form.elements["chol"].value),
    thalch: Number(form.elements["thalch"].value),
    exang: form.elements["exang"].checked,
    oldpeak: Number(form.elements["oldpeak"].value),
  };

  submitBtn.disabled = true;
  submitBtn.querySelector(".btn-label").textContent = "Assessing…";
  submitBtn.querySelector(".spinner").hidden = false;

  try {
    const res = await fetch(API.predict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }

    render(await res.json());
  } catch (err) {
    emptyState.hidden = false;
    resultBody.hidden = true;
    emptyState.innerHTML = `<h2>Could not complete the assessment</h2><p>${err.message}</p>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector(".btn-label").textContent = "Assess risk";
    submitBtn.querySelector(".spinner").hidden = true;
  }
});

/* ── render ──────────────────────────────────────────────────────── */
function render(result) {
  emptyState.hidden = true;
  resultBody.hidden = false;

  const pct = Math.round(result.probability * 100);
  document.getElementById("probValue").textContent = `${pct}%`;

  const fill = document.getElementById("gaugeFill");
  const circumference = 2 * Math.PI * 52;
  fill.style.strokeDasharray = circumference;
  fill.style.strokeDashoffset = circumference * (1 - result.probability);

  const colour = result.risk_level === "High" ? "#ff5d6c"
    : result.risk_level === "Moderate" ? "#f5b83d" : "#2dd4a7";
  fill.style.stroke = colour;

  const chip = document.getElementById("riskChip");
  chip.className = `chip ${result.risk_level.toLowerCase()}`;
  chip.textContent = `${result.risk_level} risk`;

  document.getElementById("verdictLabel").textContent = result.label;
  document.getElementById("verdictMeta").textContent =
    `${(result.confidence * 100).toFixed(1)}% confidence · ${result.model_name}`;

  // risk factors
  const factorsList = document.getElementById("factorsList");
  factorsList.innerHTML = "";
  const factorsSection = document.getElementById("factorsSection");
  if (result.risk_factors.length) {
    factorsSection.hidden = false;
    result.risk_factors.forEach((f) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="dot"></span><div>
        <span class="f-label">${f.label}:</span> <span class="f-value">${f.value}</span>
        <span class="f-detail">${f.detail}</span></div>`;
      factorsList.appendChild(li);
    });
  } else {
    factorsSection.hidden = true;
  }

  // drivers
  const driversList = document.getElementById("driversList");
  driversList.innerHTML = "";
  (result.top_drivers || []).forEach((d) => {
    const span = document.createElement("span");
    span.className = "pill";
    span.textContent = d;
    driversList.appendChild(span);
  });

  // recommendations
  const recsList = document.getElementById("recsList");
  recsList.innerHTML = "";
  result.recommendations.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    recsList.appendChild(li);
  });

  document.getElementById("resultFootnote").textContent =
    "Probability produced by the trained model; risk factors are computed from standard clinical thresholds. Not a diagnosis.";
}

init();
