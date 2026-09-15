/**
 * static/js/dashboard.js — NIDA Dashboard
 */
'use strict';

/* =========================================================================
   CHART.JS DEFAULTS
   ========================================================================= */
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#1e2d45';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

const C = {
  accent:  '#3b82f6',
  accent2: '#60a5fa',
  danger:  '#ef4444',
  warning: '#f59e0b',
  success: '#22c55e',
  teal:    '#14b8a6',
  purple:  '#a855f7',
  muted:   '#4b5e7e',
};

const CAT_COLORS = {
  dos:    C.danger,
  probe:  C.warning,
  r2l:    C.purple,
  u2r:    '#f87171',
  normal: C.success,
  unknown_anomaly: '#64748b',
};

/* =========================================================================
   CHART INSTANCES
   ========================================================================= */
let chartVolume, chartDonut, chartBar, chartHistogram;
let chartBar2, chartHistogram2;

// base options: maintainAspectRatio MUST be false — sizing is driven by
// the .chart-wrap container height set in CSS, not by the canvas aspect ratio.
const BASE = { responsive: true, maintainAspectRatio: false };

function showChartEmpty() {
  ['chartVolume','chartDonut','chartBar','chartHistogram'].forEach(id => {
    const wrap = document.getElementById(id)?.parentElement;
    if (!wrap) return;
    wrap.querySelector('canvas').style.display = 'none';
    if (!wrap.querySelector('.chart-empty')) {
      const div = document.createElement('div');
      div.className = 'chart-empty';
      div.textContent = 'Run Detection to populate charts';
      wrap.appendChild(div);
    }
  });
}

function showChartCanvases() {
  ['chartVolume','chartDonut','chartBar','chartHistogram'].forEach(id => {
    const canvas = document.getElementById(id);
    if (canvas) canvas.style.display = '';
    const wrap = canvas?.parentElement;
    wrap?.querySelector('.chart-empty')?.remove();
  });
}

function buildCharts(stats) {
  showChartCanvases();
  // Volume line
  if (chartVolume) chartVolume.destroy();
  chartVolume = new Chart(document.getElementById('chartVolume'), {
    type: 'line',
    data: {
      labels: stats.volume.labels.map(l => l.slice(11)),
      datasets: [{
        data: stats.volume.values,
        borderColor: C.accent,
        backgroundColor: 'rgba(59,130,246,.08)',
        borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0,
        pointHoverRadius: 4,
      }],
    },
    options: { ...BASE, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#1e2d45' } } } },
  });

  // Donut
  if (chartDonut) chartDonut.destroy();
  chartDonut = new Chart(document.getElementById('chartDonut'), {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Alerts'],
      datasets: [{ data: [stats.normal, stats.alerts],
        backgroundColor: [C.success, C.danger],
        borderWidth: 0, hoverOffset: 6 }],
    },
    options: { ...BASE, cutout: '72%',
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, padding: 14 } } } },
  });

  // Bar
  const catLabels = Object.keys(stats.category_breakdown);
  const catValues = Object.values(stats.category_breakdown);
  const catColors = catLabels.map(l => CAT_COLORS[l] || C.muted);

  if (chartBar) chartBar.destroy();
  chartBar = new Chart(document.getElementById('chartBar'), {
    type: 'bar',
    data: {
      labels: catLabels.map(l => l.toUpperCase()),
      datasets: [{ data: catValues, backgroundColor: catColors, borderRadius: 4, borderSkipped: false }],
    },
    options: { ...BASE, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#1e2d45' } } } },
  });

  // Histogram
  if (chartHistogram) chartHistogram.destroy();
  chartHistogram = new Chart(document.getElementById('chartHistogram'), {
    type: 'bar',
    data: {
      labels: stats.risk_histogram.labels,
      datasets: [{ data: stats.risk_histogram.values, borderRadius: 3, borderSkipped: false,
        backgroundColor: stats.risk_histogram.labels.map((_, i) =>
          i >= 8 ? C.danger : i >= 6 ? C.warning : i >= 4 ? C.accent : C.success),
      }],
    },
    options: { ...BASE, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#1e2d45' } } } },
  });

  // Analysis page duplicates
  if (document.getElementById('chartBar2')) {
    if (chartBar2) chartBar2.destroy();
    chartBar2 = new Chart(document.getElementById('chartBar2'), {
      type: 'bar',
      data: {
        labels: catLabels.map(l => l.toUpperCase()),
        datasets: [{ data: catValues, backgroundColor: catColors, borderRadius: 4, borderSkipped: false }],
      },
      options: { ...BASE, plugins: { legend: { display: false } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#1e2d45' } } } },
    });
  }
  if (document.getElementById('chartHistogram2')) {
    if (chartHistogram2) chartHistogram2.destroy();
    chartHistogram2 = new Chart(document.getElementById('chartHistogram2'), {
      type: 'bar',
      data: {
        labels: stats.risk_histogram.labels,
        datasets: [{ data: stats.risk_histogram.values, borderRadius: 3, borderSkipped: false,
          backgroundColor: stats.risk_histogram.labels.map((_, i) =>
            i >= 8 ? C.danger : i >= 6 ? C.warning : i >= 4 ? C.accent : C.success),
        }],
      },
      options: { ...BASE, plugins: { legend: { display: false } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: '#1e2d45' } } } },
    });
  }
}

/* =========================================================================
   STATS FETCH
   ========================================================================= */
async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    const stats = await res.json();

    document.getElementById('stat-total').textContent = stats.total.toLocaleString();
    document.getElementById('stat-alerts').textContent = stats.alerts.toLocaleString();
    document.getElementById('stat-normal').textContent = stats.normal.toLocaleString();
    document.getElementById('sb-records').textContent = `${stats.total.toLocaleString()} records processed`;

    // Nav badge
    const badge = document.getElementById('nav-alert-count');
    if (stats.alerts > 0) {
      badge.textContent = stats.alerts;
      badge.classList.add('visible');
    }

    if (stats.total === 0) {
      showChartEmpty();
      return;
    }

    buildCharts(stats);
  } catch (e) {
    console.error('fetchStats failed', e);
  }
}

/* =========================================================================
   COMPARISON PANEL
   ========================================================================= */
async function fetchComparison() {
  const container = document.getElementById('comparison-content');
  try {
    const res = await fetch('/api/threshold-comparison');
    const data = await res.json();

    if (data.error) {
      container.innerHTML = `<div style="padding:24px;color:var(--text-2)">${esc(data.error)}</div>`;
      return;
    }

    // Stat card
    const fpPct = data.fp_reduction_pct;
    const f1Delta = (data.f1_delta != null) ? Number(data.f1_delta) : null;
    document.getElementById('stat-fp-reduction').textContent = Math.abs(fpPct) + '%';

    if (f1Delta != null) {
      const sign = f1Delta >= 0 ? '+' : '';
      const col  = f1Delta >= 0 ? 'var(--success)' : 'var(--warning)';
      document.getElementById('stat-fp-label').innerHTML =
        `FP Reduction vs Naive <span style="color:${col};font-size:10px">(F1 ${sign}${f1Delta.toFixed(3)})</span>`;
    }

    function fmtDelta(v) {
      if (v == null) return 'N/A';
      const n = Number(v);
      return (n >= 0 ? '+' : '') + n.toFixed(3);
    }

    // Banner
    const bannerClass = data.beats_naive_on_both ? 'win' : 'tradeoff';
    const bannerIcon  = data.beats_naive_on_both ? '✅' : '⚠';
    const bannerText  = data.beats_naive_on_both
      ? `${bannerIcon} ${Math.abs(fpPct)}% fewer false positives · F1 ${fmtDelta(f1Delta)} vs naive · threshold ${data.risk_threshold_used}/100`
      : `${bannerIcon} ${Math.abs(fpPct)}% fewer FPs · F1 ${fmtDelta(f1Delta)} vs naive · threshold ${data.risk_threshold_used}/100`;
    const bannerNote  = data.trade_off_note
      ? `<span class="banner-note">${esc(data.trade_off_note)}</span>` : '';

    // Grid table
    let gridHtml = '';
    if (data.grid_results && data.grid_results.length > 0) {
      const nFP = data.naive.fp;
      const nF1 = data.naive.f1;
      const gridRows = data.grid_results.map(r => {
        const dFP  = r.fp - nFP;
        const dF1  = r.f1 - nF1;
        const chosen = r.threshold === data.risk_threshold_used;
        const fpCol = dFP < 0 ? `color:var(--success)` : dFP > 0 ? `color:var(--danger)` : '';
        const f1Col = dF1 >= 0 ? `color:var(--success)` : `color:var(--warning)`;
        return `<tr class="${chosen ? 'chosen-row' : ''}">
          <td>${r.threshold}${chosen ? ' ◀' : ''}</td>
          <td>${r.fp.toLocaleString()}</td>
          <td>${r.tp.toLocaleString()}</td>
          <td>${r.precision.toFixed(3)}</td>
          <td>${r.recall.toFixed(3)}</td>
          <td>${r.f1.toFixed(3)}</td>
          <td style="${fpCol}">${dFP >= 0 ? '+' : ''}${dFP.toLocaleString()}</td>
          <td style="${f1Col}">${dF1 >= 0 ? '+' : ''}${dF1.toFixed(3)}</td>
        </tr>`;
      }).join('');

      gridHtml = `
        <div class="grid-table-wrap">
          <div class="grid-table-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
            <span>📊 Full grid search — thresholds 30 to 70, step 5 (click to expand)</span>
            <span style="font-size:10px;color:var(--text-3)">▼</span>
          </div>
          <div class="grid-table-body hidden">
            <table class="grid-table">
              <thead><tr>
                <th>Threshold</th><th>FP</th><th>TP</th>
                <th>Precision</th><th>Recall</th><th>F1</th>
                <th>ΔFP</th><th>ΔF1</th>
              </tr></thead>
              <tbody>${gridRows}</tbody>
            </table>
            <div style="padding:8px 14px;font-size:10.5px;color:var(--text-3);font-family:var(--mono)">
              Full grid also saved to results/threshold_grid.csv
            </div>
          </div>
        </div>`;
    }

    container.innerHTML = `
      <div class="comparison-wrap">
        <div class="result-banner ${bannerClass}">${bannerText}${bannerNote}</div>
        <div class="comparison-grid">
          <div class="comparison-col is-worse">
            <h4>🔴 Naïve Threshold (baseline)</h4>
            ${cmRow('False Positives', data.naive.fp.toLocaleString())}
            ${cmRow('True Positives',  data.naive.tp.toLocaleString())}
            ${cmRow('Precision', data.naive.precision.toFixed(3))}
            ${cmRow('Recall',    data.naive.recall.toFixed(3))}
            ${cmRow('F1 Score',  data.naive.f1.toFixed(3))}
          </div>
          <div class="comparison-col ${data.beats_naive_on_both ? 'is-better' : 'is-tradeoff'}">
            <h4>${data.beats_naive_on_both ? '✅' : '⚖'} Weighted Risk Score (t=${data.risk_threshold_used})</h4>
            ${cmRow('False Positives', data.risk_score.fp.toLocaleString())}
            ${cmRow('True Positives',  data.risk_score.tp.toLocaleString())}
            ${cmRow('Precision', data.risk_score.precision.toFixed(3))}
            ${cmRow('Recall',    data.risk_score.recall.toFixed(3))}
            ${cmRow('F1 Score',  data.risk_score.f1.toFixed(3))}
          </div>
        </div>
        ${gridHtml}
      </div>`;
  } catch (e) {
    container.innerHTML = `<div style="padding:24px;color:var(--text-2)">Comparison unavailable (models may not be trained).</div>`;
  }
}

function cmRow(label, val) {
  return `<div class="comparison-metric"><span class="metric-label">${label}</span><span class="metric-val">${val}</span></div>`;
}

/* =========================================================================
   ALERTS
   ========================================================================= */
let allAlerts = [];
let filterCat = '', filterMinRisk = 0;

function riskClass(s) {
  if (s >= 80) return 'risk-critical';
  if (s >= 60) return 'risk-high';
  if (s >= 40) return 'risk-medium';
  return 'risk-low';
}

function rowSeverityClass(s) {
  if (s >= 80) return 'row-critical';
  if (s >= 60) return 'row-high';
  return '';
}

function catPill(cat) {
  const label = cat === 'UNKNOWN_ANOMALY' ? 'UNKNOWN' : (cat || '').toUpperCase();
  const cls   = 'cat-' + (cat === 'UNKNOWN_ANOMALY' ? 'unknown' : (cat || 'unknown').toLowerCase());
  return `<span class="cat-pill ${cls}">${label}</span>`;
}

function renderAlerts() {
  const tbody = document.getElementById('alerts-tbody');
  const filtered = allAlerts.filter(a =>
    (!filterCat || a.attack_category === filterCat || a.attack_category?.toLowerCase() === filterCat) &&
    (a.risk_score >= filterMinRisk)
  );

  if (!filtered.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No alerts match the current filters.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered.map(a => {
    const isAnomaly = a.attack_category === 'UNKNOWN_ANOMALY';
    const mitreCell = a.technique_id
      ? `<span class="mitre-badge">${esc(a.technique_id)}</span>
         <div class="mitre-name" style="margin-top:3px">${esc(a.technique_name || '')}</div>`
      : `<span style="font-size:11px;color:var(--text-3);font-style:italic">No mapping${isAnomaly ? ' (anomaly-only)' : ''}</span>`;

    const sublabel = isAnomaly
      ? `<div style="font-size:10px;color:var(--text-3);margin-top:2px">clf: normal · outlier flagged</div>`
      : a.attack_label ? `<div style="font-size:10px;color:var(--text-3);margin-top:2px">${esc(a.attack_label)}</div>` : '';

    const summaryText = a.incident_summary
      ? esc(a.incident_summary.slice(0, 90)) + (a.incident_summary.length > 90 ? '…' : '')
      : a.reason_string ? esc(a.reason_string.slice(0, 90)) + '…' : '—';

    return `<tr class="${rowSeverityClass(a.risk_score)}">
      <td class="mono" style="font-size:11px;color:var(--text-2);white-space:nowrap">${a.timestamp.slice(0,19).replace('T',' ')}</td>
      <td>${catPill(a.attack_category)}${sublabel}</td>
      <td><span class="risk-pill ${riskClass(a.risk_score)}">${Math.round(a.risk_score)}</span></td>
      <td>${mitreCell}</td>
      <td style="max-width:260px;font-size:11.5px;color:var(--text-2)" title="${esc(a.incident_summary||'')}">${summaryText}</td>
      <td><button class="expand-btn" onclick="openModal(${a.id})">View</button></td>
    </tr>`;
  }).join('');
}

async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts?limit=100');
    allAlerts = await res.json();
    renderAlerts();
  } catch (e) {
    console.error('fetchAlerts failed', e);
  }
}

/* =========================================================================
   MODAL
   ========================================================================= */
async function openModal(alertId) {
  document.getElementById('alert-modal').classList.remove('hidden');
  const body = document.getElementById('modal-body');
  body.innerHTML = '<div style="padding:30px;text-align:center;color:var(--text-2)">Loading…</div>';

  try {
    const res = await fetch(`/api/alert/${alertId}`);
    const a = await res.json();

    const isAnomaly = (a.attack_category || '').toUpperCase() === 'UNKNOWN_ANOMALY';

    // KPI row
    const kpiHtml = `
      <div class="modal-kpi-row">
        <div class="modal-kpi">
          <span class="modal-kpi-label">Risk Score</span>
          <span class="modal-kpi-val" style="${a.risk_score>=80?'color:var(--danger)':a.risk_score>=60?'color:var(--warning)':'color:var(--accent2)'}">${Math.round(a.risk_score)}/100</span>
        </div>
        <div class="modal-kpi">
          <span class="modal-kpi-label">Category</span>
          <span class="modal-kpi-val">${catPill(a.attack_category)}</span>
        </div>
        ${a.technique_id ? `<div class="modal-kpi">
          <span class="modal-kpi-label">MITRE</span>
          <span class="modal-kpi-val"><span class="mitre-badge">${esc(a.technique_id)}</span></span>
        </div>` : ''}
        <div class="modal-kpi">
          <span class="modal-kpi-label">Classifier P(attack)</span>
          <span class="modal-kpi-val">${((a.clf_confidence||0)*100).toFixed(1)}%</span>
        </div>
        <div class="modal-kpi">
          <span class="modal-kpi-label">Anomaly Score</span>
          <span class="modal-kpi-val">${((a.anomaly_score||0)*100).toFixed(1)}%</span>
        </div>
      </div>`;

    // Summary
    const summaryHtml = `
      <div class="modal-section">
        <div class="modal-section-title">📝 Incident Summary</div>
        <div class="summary-block">${esc(a.incident_summary || 'No summary generated.')}</div>
        <span class="summary-mode-tag">${
          a.summary_mode === 'llm' ? '✨ LLM-generated' :
          a.summary_mode === 'cache' ? '📦 Cached' : '📄 Template (offline)'
        }</span>
      </div>`;

    // Features
    const featRows = (a.top_features || []).map(f => `
      <div class="feature-row">
        <span class="feature-name">${esc(f.name || '—')}</span>
        <span class="feature-val">${typeof f.raw_value === 'number' ? f.raw_value.toFixed(3) : esc(String(f.raw_value ?? '—'))}</span>
        <span class="feature-avg">avg ${typeof f.normal_mean === 'number' ? f.normal_mean.toFixed(2) : '—'}</span>
        <span class="feature-shap ${typeof f.shap_value === 'number' && f.shap_value >= 0 ? 'shap-pos' : 'shap-neg'}">
          ${typeof f.shap_value === 'number' ? (f.shap_value >= 0 ? '+' : '') + f.shap_value.toFixed(3) : '—'}
        </span>
      </div>`).join('') || '<div style="padding:10px;color:var(--text-3);font-size:12px">No feature data available.</div>';

    const featHtml = `
      <div class="modal-section">
        <div class="modal-section-title">🔬 Contributing Features</div>
        <p style="font-size:12px;color:var(--text-2);margin-bottom:8px">${esc(a.reason_string || '—')}</p>
        <div class="feature-grid">${featRows}</div>
      </div>`;

    // MITRE
    const mitreHtml = `
      <div class="modal-section">
        <div class="modal-section-title">🎯 MITRE ATT&CK</div>
        <div class="mitre-block ${a.technique_id ? '' : 'no-mitre'}">
          ${a.technique_id
            ? esc(a.technique_reasoning || 'No reasoning available.')
            : isAnomaly
              ? 'No MITRE technique mapped — the primary classifier labelled this record normal. MITRE tags are only assigned to confirmed attack categories.'
              : 'Technique not mapped for this record.'}
        </div>
      </div>`;

    // Similarity
    const simHtml = `
      <div class="modal-section">
        <div class="modal-section-title">📅 Historical Similarity</div>
        ${a.nearest_label && a.nearest_label !== 'null'
          ? `<div class="similarity-row">Most similar historical incident: <strong>${esc(a.nearest_label)}</strong>
             &ensp;·&ensp; ${Math.round((a.nearest_similarity || 0) * 100)}% feature similarity</div>`
          : `<div class="similarity-row" style="color:var(--text-3);font-style:italic">Similarity index not built yet.</div>`}
      </div>`;

    body.innerHTML = `
      <h3 style="font-size:15px;margin-bottom:4px;font-weight:700">Alert #${a.id}</h3>
      <p style="font-size:11px;color:var(--text-3);font-family:var(--mono);margin-bottom:18px">${a.timestamp}</p>
      ${kpiHtml}
      ${summaryHtml}
      ${featHtml}
      ${mitreHtml}
      ${simHtml}`;
  } catch (e) {
    body.innerHTML = `<div style="padding:20px;color:var(--danger)">Failed to load: ${esc(e.message)}</div>`;
  }
}

function closeModal() {
  document.getElementById('alert-modal').classList.add('hidden');
}

/* =========================================================================
   SIDEBAR NAVIGATION
   ========================================================================= */
function navTo(pageId, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const page = document.getElementById('page-' + pageId);
  if (page) page.classList.add('active');
  if (el) el.classList.add('active');
  // Scroll alerts table into view when switching to alerts page
  if (pageId === 'alerts') fetchAlerts();
  return false;
}

/* =========================================================================
   SIMULATE BUTTON
   ========================================================================= */
document.getElementById('btn-simulate').addEventListener('click', async () => {
  const btn    = document.getElementById('btn-simulate');
  const status = document.getElementById('sim-status');
  btn.disabled = true;
  status.textContent = '⏳ Running…';

  try {
    const res  = await fetch('/api/simulate?n=200', { method: 'POST' });
    const data = await res.json();
    if (data.error) {
      status.textContent = '❌ ' + data.error;
    } else {
      status.textContent = `✅ ${data.processed} processed · ${data.new_alerts} new alerts`;
      await fetchStats();
      await fetchAlerts();
    }
  } catch (e) {
    status.textContent = '❌ ' + e.message;
  } finally {
    btn.disabled = false;
  }
});

/* =========================================================================
   REFRESH BUTTON
   ========================================================================= */
document.getElementById('btn-refresh').addEventListener('click', async () => {
  const btn = document.getElementById('btn-refresh');
  btn.classList.add('refreshing');
  await fetchStats();
  await fetchAlerts();
  btn.classList.remove('refreshing');
});

/* =========================================================================
   FILTERS
   ========================================================================= */
document.getElementById('filter-category').addEventListener('change', e => {
  filterCat = e.target.value;
  renderAlerts();
});
document.getElementById('filter-risk').addEventListener('input', e => {
  filterMinRisk = parseInt(e.target.value, 10);
  document.getElementById('filter-risk-val').textContent = filterMinRisk;
  renderAlerts();
});

/* =========================================================================
   LIVE CLOCK
   ========================================================================= */
function tick() {
  const el = document.getElementById('sb-clock');
  if (el) el.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
}
tick();
setInterval(tick, 1000);

/* =========================================================================
   KEYBOARD
   ========================================================================= */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

/* =========================================================================
   HELPERS
   ========================================================================= */
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* =========================================================================
   PACKET CAPTURE CONTROLS
   ========================================================================= */
let capturePoller = null;

async function updateCaptureStatus() {
  try {
    const res  = await fetch('/api/capture/status');
    const data = await res.json();
    const dot   = document.getElementById('cap-dot');
    const label = document.getElementById('cap-label');
    const stats = document.getElementById('cap-stats');
    const startBtn = document.getElementById('btn-cap-start');
    const stopBtn  = document.getElementById('btn-cap-stop');

    if (data.running) {
      dot.style.background = 'var(--success)';
      dot.classList.add('pulse');
      label.textContent = data.backend ? `Live · ${data.backend}` : 'Running';
      startBtn.style.display = 'none';
      stopBtn.style.display  = 'block';
      stats.textContent = `${data.packets_seen} pkts · ${data.alerts_generated} alerts`;
    } else {
      dot.style.background = data.error ? 'var(--danger)' : 'var(--text-3)';
      dot.classList.remove('pulse');
      label.textContent = data.error ? 'Error' : 'Stopped';
      startBtn.style.display = 'block';
      stopBtn.style.display  = 'none';
      stats.textContent = data.error ? data.error : '';
      if (capturePoller) { clearInterval(capturePoller); capturePoller = null; }
    }
  } catch (_) {}
}

document.getElementById('btn-cap-start').addEventListener('click', async () => {
  const iface = document.getElementById('cap-iface').value.trim();
  await fetch('/api/capture/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ interface: iface }),
  });
  await updateCaptureStatus();
  // Poll every 3s while running
  if (!capturePoller) capturePoller = setInterval(updateCaptureStatus, 3000);
});

document.getElementById('btn-cap-stop').addEventListener('click', async () => {
  await fetch('/api/capture/stop', { method: 'POST' });
  await updateCaptureStatus();
  if (capturePoller) { clearInterval(capturePoller); capturePoller = null; }
});

/* =========================================================================
   INIT
   ========================================================================= */
(async () => {
  await fetchStats();
  await fetchAlerts();
  await fetchComparison();
  await updateCaptureStatus();
})();
