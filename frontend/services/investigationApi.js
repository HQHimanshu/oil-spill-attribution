/**
 * OceanGuard AI - Frontend API Service (Phase 31).
 * Communicates with FastAPI backend for real SAR detection, metocean backtracking, AIS correlation,
 * ML model evaluation metrics, and grounded RAG Investigation Copilot queries.
 */

const API_CANDIDATES = [
  window.__APP_API_BASE__,
  'http://127.0.0.1:8000',
  'http://localhost:8000',
  'http://127.0.0.1:8001',
  'http://localhost:8001'
].filter(Boolean);

async function requestJson(path, options = {}) {
  let lastError;

  for (const base of API_CANDIDATES) {
    try {
      const requestUrl = `${base}${path.startsWith('/') ? path : `/${path}`}`;
      const response = await fetch(requestUrl, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
      });

      if (!response.ok) {
        const text = await response.text();
        lastError = new Error(text || `HTTP ${response.status}`);
        continue;
      }

      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error('Backend API unavailable');
}

function normalizeInvestigation(raw) {
  const vessels = Array.isArray(raw?.vessels) ? raw.vessels : [];
  const selected = raw?.selectedVessel || vessels[0] || null;

  return {
    ...raw,
    vessels,
    selectedVessel: selected,
    spill: {
      ...raw?.spill,
      centroid: raw?.spill?.centroid || { latitude: 28.582, longitude: -94.925 },
      boundary: Array.isArray(raw?.spill?.boundary) ? raw.spill.boundary : []
    },
    origin: {
      ...raw?.origin,
      probableOrigin: raw?.origin?.probableOrigin || { latitude: 28.22, longitude: -95.40 },
      route: Array.isArray(raw?.origin?.route) ? raw.origin.route : []
    }
  };
}

window.InvestigationApi = {
  async createInvestigation(payload) {
    const result = await requestJson('/api/investigations', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    return normalizeInvestigation(result);
  },

  async getInvestigation(id) {
    const result = await requestJson(`/api/investigations/${encodeURIComponent(id)}`);
    return normalizeInvestigation(result);
  },

  async analyzeInvestigation(id) {
    const result = await requestJson(`/api/investigations/${encodeURIComponent(id)}/analyze`, {
      method: 'POST',
    });
    return normalizeInvestigation(result);
  },

  async askCopilot(id, question) {
    return requestJson(`/api/investigations/${encodeURIComponent(id)}/ask`, {
      method: 'POST',
      body: JSON.stringify({ question }),
    });
  },

  async getModelMetrics() {
    return requestJson('/api/model/metrics');
  },

  async getProvenance(id) {
    const query = id ? `?investigation_id=${encodeURIComponent(id)}` : '';
    return requestJson(`/api/provenance${query}`);
  },

  async getSarScenes() {
    return requestJson('/api/sar/scenes');
  },

  async getPalsarScenes(limit = 40) {
    return requestJson(`/api/palsar/scenes?limit=${encodeURIComponent(limit)}`);
  },

  async searchLocations(query) {
    return requestJson(`/api/location/search?query=${encodeURIComponent(query)}`);
  },

  async getLiveEnvironmental(latitude, longitude, timestamp) {
    const params = new URLSearchParams({ latitude, longitude });
    if (timestamp) params.append('timestamp', timestamp);
    return requestJson(`/api/environmental/live?${params.toString()}`);
  },

  async triggerTraining() {
    return requestJson('/api/model/train', {
      method: 'POST'
    });
  },

  async detectSpill(file, latitude, longitude) {
    const formData = new FormData();
    formData.append('file', file);
    if (latitude) formData.append('latitude', String(latitude));
    if (longitude) formData.append('longitude', String(longitude));

    for (const base of API_CANDIDATES) {
      try {
        const response = await fetch(`${base}/detect-spill`, {
          method: 'POST',
          body: formData
        });
        if (response.ok) return await response.json();
      } catch (err) {
        // try next
      }
    }
    throw new Error('Image detection failed');
  }
};

