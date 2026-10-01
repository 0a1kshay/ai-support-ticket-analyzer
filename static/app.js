
/* AI Support Ticket Analyzer — Frontend Application */

const API = '';   // same origin

// ── Utilities ──────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function badge(text, cls) {
  return `<span class="badge badge-${cls}">${text}</span>`;
}

function escHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function showLoading(containerId, msg = 'Loading...') {
  $(containerId).innerHTML =
    `<div class="loading"><span class="spinner"></span>${escHtml(msg)}</div>`;
}

function showError(containerId, msg) {
  $(containerId).innerHTML =
    `<div class="error-msg">&#x26A0; ${escHtml(msg)}</div>`;
}

// ── Health / LLM badge ─────────────────────────────────────

async function loadHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const badge = $('llm-badge');
    if (d.llm_available) {
      badge.textContent = `✓ ${d.llm_provider}`;
      badge.className = 'available';
    } else {
      badge.textContent = `✗ LLM Unavailable`;
      badge.className = 'unavailable';
      badge.title = d.message;
    }
  } catch {
    $('llm-badge').textContent = '✗ API Offline';
    $('llm-badge').className = 'unavailable';
  }
}

// ── Stats / KPIs ───────────────────────────────────────────

async function loadStats() {
  try {
    const r = await fetch(`${API}/api/stats`);
    const d = await r.json();
    $('kpi-total').textContent     = d.total_tickets ?? '-';
    $('kpi-open').textContent      = d.open_tickets ?? '-';
    $('kpi-resolved').textContent  = d.resolved_tickets ?? '-';
    $('kpi-escalated').textContent = d.escalated_tickets ?? '-';
    $('kpi-critical').textContent  = d.critical_tickets ?? '-';
    $('kpi-resp').textContent      = d.avg_response_time_hrs != null ? d.avg_response_time_hrs + 'h' : '-';
    $('kpi-resol').textContent     = d.avg_resolution_time_hrs != null ? d.avg_resolution_time_hrs + 'h' : '-';
    $('kpi-rating').textContent    = d.avg_customer_rating != null ? d.avg_customer_rating + '/5' : '-';
    $('kpi-anomalies').textContent = d.anomaly_count ?? '-';
  } catch (e) {
    console.error('Stats load failed:', e);
  }
}

// ── Anomalies ──────────────────────────────────────────────

async function loadAnomalies() {
  showLoading('anomaly-container', 'Detecting anomalies...');
  try {
    const r = await fetch(`${API}/api/anomalies`);
    const d = await r.json();
    renderAnomalies(d);
  } catch (e) {
    showError('anomaly-container', 'Failed to load anomalies: ' + e.message);
  }
}

function renderAnomalies(d) {
  // Summary stats
  const meta = $('anomaly-meta');
  const byType = d.by_type || {};
  const bySev = d.by_severity || {};
  meta.innerHTML = `
    <div class="anomaly-stat"><strong>${d.total_count}</strong>Total Anomalies</div>
    ${Object.entries(bySev).map(([sev, cnt]) =>
      `<div class="anomaly-stat"><strong>${cnt}</strong>${sev}</div>`
    ).join('')}
    ${Object.entries(byType).map(([typ, cnt]) =>
      `<div class="anomaly-stat"><strong>${cnt}</strong>${typ.replace(/_/g,' ')}</div>`
    ).join('')}
  `;

  const container = $('anomaly-container');
  if (!d.anomalies || d.anomalies.length === 0) {
    container.innerHTML = '<div class="empty-state">No anomalies detected.</div>';
    return;
  }

  container.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Ticket ID</th>
            <th>Type</th>
            <th>Severity</th>
            <th>Value</th>
            <th>Threshold</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          ${d.anomalies.map(a => `
            <tr>
              <td><code>${escHtml(a.ticket_id)}</code></td>
              <td>${escHtml(a.anomaly_type.replace(/_/g,' '))}</td>
              <td>${badge(a.severity, a.severity)}</td>
              <td>${a.metric_value != null ? a.metric_value : '-'}</td>
              <td>${a.threshold != null ? a.threshold : '-'}</td>
              <td style="max-width:340px;font-size:0.8rem;color:var(--text-muted)">${escHtml(a.reason)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

// ── Query ──────────────────────────────────────────────────

function setQuestion(btn) {
  $('question-input').value = btn.textContent.trim();
  $('question-input').focus();
}

async function submitQuery() {
  const question = $('question-input').value.trim();
  if (!question) {
    showError('query-status', 'Please enter a question.');
    return;
  }

  const btn = $('ask-btn');
  btn.disabled = true;
  btn.textContent = 'Thinking...';
  $('query-status').innerHTML = '';
  $('answer-section').style.display = 'none';

  try {
    const r = await fetch(`${API}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }

    const d = await r.json();
    renderAnswer(d);
    $('query-status').innerHTML = '';

  } catch (e) {
    showError('query-status', 'Query failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Ask AI';
  }
}

function renderAnswer(d) {
  const section = $('answer-section');
  section.style.display = 'block';

  $('answer-text').textContent = d.answer || 'No answer.';
  $('intent-tag').textContent = d.intent || '';

  const data = d.data || {};
  const container = $('results-container');
  const countTag = $('count-tag');

  // If there are rows, show a table
  if (data.rows && data.rows.length > 0) {
    countTag.textContent = `${data.rows.length} row(s)`;
    container.innerHTML = renderTable(data.rows);
  } else if (data.full_ranking && data.full_ranking.length > 0) {
    countTag.textContent = `${data.full_ranking.length} items`;
    container.innerHTML = renderTable(data.full_ranking);
  } else if (data.count !== undefined) {
    countTag.textContent = `Count: ${data.count}`;
    container.innerHTML = '';
  } else if (data.average !== undefined) {
    countTag.textContent = `Sample: ${data.sample_size || '-'}`;
    container.innerHTML = '';
  } else if (data.sample_anomalies) {
    countTag.textContent = `${data.total_anomalies} anomalies`;
    container.innerHTML = data.sample_anomalies.length > 0
      ? renderTable(data.sample_anomalies)
      : '';
  } else {
    countTag.textContent = '';
    container.innerHTML = '';
  }

  // Scroll to answer
  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderTable(rows) {
  if (!rows || rows.length === 0) return '<div class="empty-state">No results.</div>';
  const cols = Object.keys(rows[0]);
  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${cols.map(c => `<th>${escHtml(c)}</th>`).join('')}</tr></thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              ${cols.map(c => {
                const v = row[c];
                if (c === 'priority') return `<td>${badge(v, v)}</td>`;
                if (c === 'status') return `<td>${badge(v, v)}</td>`;
                if (c === 'severity') return `<td>${badge(v, v)}</td>`;
                if (c === 'anomaly_type') return `<td style="font-size:0.8rem">${escHtml(String(v||'').replace(/_/g,' '))}</td>`;
                if (c === 'reason') return `<td style="max-width:300px;font-size:0.78rem;color:var(--text-muted)">${escHtml(v)}</td>`;
                if (c === 'issue_summary') return `<td style="max-width:260px;font-size:0.8rem;color:var(--text-muted)">${escHtml(v)}</td>`;
                return `<td>${escHtml(v != null ? String(v) : '-')}</td>`;
              }).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

// ── Keyboard shortcut ──────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  $('question-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') submitQuery();
  });
  loadHealth();
  loadStats();
  loadAnomalies();
});
