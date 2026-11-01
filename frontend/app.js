const $ = (s) => document.querySelector(s);
const NUMERIC = ["temperature_C", "pressure_bar", "vibration_mm_s",
  "rotational_speed_rpm", "production_speed_units_hr", "tool_wear_min"];
const label = (f) => f.replace(/_/g, " ");
let cfg, timer;

const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
};

document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== b.dataset.tab));
  if (b.dataset.tab === "alerts") loadAlerts();
});

async function init() {
  cfg = await api("/api/config");
  const m = await api("/api/metrics");
  $("#model").textContent = `${m.selected} · hold-out ROC-AUC ${m.holdout.roc_auc.toFixed(2)}`;
  $("#machine").innerHTML = cfg.machines.map((x) => `<option>${x}</option>`).join("");
  $("#machine").onchange = schedule;
  $("#sliders").innerHTML = NUMERIC.map((f) => {
    const [lo, hi] = cfg.ranges[f];
    const step = (hi - lo) / 200;
    return `<div class="slider"><div><span>${label(f)} (${cfg.units[f]})</span><b id="v_${f}"></b></div>
      <input type="range" id="${f}" min="${lo}" max="${hi}" step="${step}" value="${cfg.nominal[f]}"></div>`;
  }).join("");
  NUMERIC.forEach((f) => {
    $("#" + f).oninput = schedule;
    $("#" + f).onchange = () => run(true);  // log alert only when the slider is released
  });
  $("#machine").onchange = () => run(true);
  schedule();
}

window.addEventListener("unhandledrejection", (e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="background:#d63b3b;color:#fff;padding:8px 24px">Cannot reach the backend (${e.reason.message}).
     Start it with <code>uvicorn backend:app</code> and open http://127.0.0.1:8000 (not the HTML file directly).</div>`);
}, { once: true });

function schedule() { clearTimeout(timer); timer = setTimeout(() => run(false), 150); }

async function run(log) {
  const body = { machine_id: $("#machine").value };
  NUMERIC.forEach((f) => { body[f] = +$("#" + f).value; $("#v_" + f).textContent = (+$("#" + f).value).toFixed(1); });
  const r = await api("/api/predict" + (log ? "?log=true" : ""), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  $("#prob").textContent = (r.probability * 100).toFixed(1) + "%";
  const lv = $("#level"); lv.textContent = r.level; lv.className = "level " + r.level;
  $("#actions").innerHTML = r.actions.length ? r.actions.map((a) => `<li>${a}</li>`).join("") : "<li>No action needed</li>";
  const max = Math.max(...r.causes.map((c) => Math.abs(c.risk_contribution)), 0.01);
  $("#causes").innerHTML = r.causes.map((c) => `<div class="bar"><span>${label(c.feature)}</span>
    <i style="width:${Math.max(0, c.risk_contribution) / max * 100}%"></i>
    <span>${(c.risk_contribution * 100).toFixed(1)}%</span></div>`).join("");
  refreshBadge();
}

function renderBatch(d) {
  $("#summary").innerHTML = `<span>${d.total} readings</span>` +
    ["HIGH", "MEDIUM", "LOW"].map((k) => `<span class="pill ${k}">${k}: ${d.summary[k] || 0}</span>`).join("");
  if (!d.rows.length) return;
  const cols = Object.keys(d.rows[0]);
  $("#table").innerHTML = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>" +
    d.rows.map((r) => `<tr class="${r.risk}">` + cols.map((c) => `<td>${r[c]}</td>`).join("") + "</tr>").join("");
}

$("#file").onchange = async (e) => {
  const fd = new FormData(); fd.append("file", e.target.files[0]);
  try { renderBatch(await api("/api/batch", { method: "POST", body: fd })); }
  catch (err) { alert(err.message); }
};
$("#sample").onclick = async () => renderBatch(await api("/api/sample"));

async function loadAlerts() {
  const a = await api("/api/alerts");
  $("#alertlist").innerHTML = a.length ? a.map((x) => `<div class="alert"><b>${x.time} · ${x.machine}</b> —
    ${(x.probability * 100).toFixed(0)}% defect risk, main cause: ${label(x.top_cause)}<br>
    ${x.actions.join("; ")}</div>`).join("") : "<p class='muted'>No alerts yet. Raise a parameter in Live Check.</p>";
}
async function refreshBadge() {
  const n = (await api("/api/alerts")).length, b = $("#badge");
  b.textContent = n; b.classList.toggle("hidden", !n);
}

init();
