const API_BASE = (window.__APP_API_BASE__ || 'http://127.0.0.1:8000');

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Request failed');
  }

  return response.json();
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
      centroid: raw?.spill?.centroid || { latitude: 0, longitude: 0 },
    },
  };
}

window.InvestigationApi = {
  async createInvestigation(payload) {
    try {
      const result = await requestJson(`${API_BASE}/api/investigations`, {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      return normalizeInvestigation(result);
    } catch (error) {
      const fallback = window.InvestigationMockData?.buildMockInvestigation(payload);
      if (fallback) return normalizeInvestigation(fallback);
      throw error;
    }
  },

  async getInvestigation(id) {
    try {
      const result = await requestJson(`${API_BASE}/api/investigations/${encodeURIComponent(id)}`);
      return normalizeInvestigation(result);
    } catch (error) {
      const fallback = window.InvestigationMockData?.buildMockInvestigation();
      if (fallback) return normalizeInvestigation(fallback);
      throw error;
    }
  },

  async analyzeInvestigation(id) {
    try {
      const result = await requestJson(`${API_BASE}/api/investigations/${encodeURIComponent(id)}/analyze`, {
        method: 'POST',
      });
      return normalizeInvestigation(result);
    } catch (error) {
      const fallback = window.InvestigationMockData?.buildMockInvestigation();
      if (fallback) return normalizeInvestigation(fallback);
      throw error;
    }
  },

  async getVessels(id) {
    try {
      const result = await requestJson(`${API_BASE}/api/investigations/${encodeURIComponent(id)}/vessels`);
      return result;
    } catch (error) {
      const fallback = window.InvestigationMockData?.buildMockInvestigation();
      return fallback?.vessels || [];
    }
  },

  async getVessel(id, vesselId) {
    try {
      const result = await requestJson(`${API_BASE}/api/investigations/${encodeURIComponent(id)}/vessels/${encodeURIComponent(vesselId)}`);
      return result;
    } catch (error) {
      const fallback = window.InvestigationMockData?.buildMockInvestigation();
      const vessel = (fallback?.vessels || []).find((item) => String(item.id) === String(vesselId));
      return vessel || fallback?.selectedVessel || {};
    }
  },
};
