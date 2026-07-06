/**
 * RAG Learning Simulator — API Helper
 * Centralized fetch() wrapper for all FastAPI endpoints
 */

const API = {
  BASE: '',

  async post(endpoint, data) {
    const res = await fetch(`${this.BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'API error');
    }
    return res.json();
  },

  async get(endpoint) {
    const res = await fetch(`${this.BASE}${endpoint}`);
    if (!res.ok) throw new Error(`GET ${endpoint} failed`);
    return res.json();
  },

  async upload(file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${this.BASE}/api/upload`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  chunk: (text, strategy, opts) =>
    API.post('/api/chunk', { text, strategy, ...opts }),

  embed: (chunks) =>
    API.post('/api/embed', { chunks }),

  embedQuery: (query) =>
    API.post('/api/embed-query', { query }),

  store: (chunks) =>
    API.post('/api/store', { chunks }),

  search: (query_embedding, top_k = 5) =>
    API.post('/api/search', { query_embedding, top_k }),

  rerank: (query, chunks, top_k = 3) =>
    API.post('/api/rerank', { query, chunks, top_k }),

  generate: (query, context_chunks) =>
    API.post('/api/generate', { query, context_chunks }),

  dbStats: () =>
    API.get('/api/db-stats'),
};

/**
 * UI Helpers
 */
function showAlert(containerId, message, type = 'info') {
  const el = document.getElementById(containerId);
  if (!el) return;
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  el.innerHTML = `<div class="alert alert-${type}">${icons[type]} ${message}</div>`;
}

function setLoading(btnId, loading, text = 'Run') {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  btn.innerHTML = loading
    ? `<span class="spinner"></span> Processing...`
    : text;
}

function typewriterEffect(elementId, text, speed = 25) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.classList.add('streaming');
  el.textContent = '';
  let i = 0;
  const interval = setInterval(() => {
    if (i < text.length) {
      el.textContent += text[i++];
    } else {
      el.classList.remove('streaming');
      clearInterval(interval);
    }
  }, speed);
}

function renderEmbeddingBar(containerId, embedding, count = 32) {
  const el = document.getElementById(containerId);
  if (!el || !embedding) return;
  const sample = embedding.slice(0, count);
  const max = Math.max(...sample.map(Math.abs)) || 1;
  el.innerHTML = `<div class="embedding-bar">
    ${sample.map(v => {
      const pct = Math.round((Math.abs(v) / max) * 100);
      const cls = v >= 0 ? 'bar-positive' : 'bar-negative';
      return `<div class="bar-cell ${cls}" style="height:${Math.max(pct, 4)}%" title="${v.toFixed(4)}"></div>`;
    }).join('')}
  </div>`;
}

function scoreClass(score) {
  if (score >= 0.7) return 'score-high';
  if (score >= 0.4) return 'score-mid';
  return 'score-low';
}

// Persist pipeline data across steps via sessionStorage
const PipelineState = {
  set(key, val) { sessionStorage.setItem(`rag_${key}`, JSON.stringify(val)); },
  get(key) {
    try { return JSON.parse(sessionStorage.getItem(`rag_${key}`) || 'null'); }
    catch { return null; }
  },
  clear() {
    ['text', 'chunks', 'embedded_chunks', 'query', 'query_embedding',
     'retrieved', 'reranked', 'response'].forEach(k => sessionStorage.removeItem(`rag_${k}`));
  }
};
