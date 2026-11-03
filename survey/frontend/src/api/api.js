async function get(path, options = {}) {
    const response = await fetch(path, {
        headers: { 'content-type': 'application/json' },
        ...options
    });

    if (!response.ok) {
        const body = await response.json().catch(_ => null);
        const detail = typeof body?.detail === 'string' ? body.detail : response.statusText;
        const error = new Error(`${response.status} ${detail}`);
        error.status = response.status;
        throw error;
    }

    return response.json();
}

const post = (path, body) => get(path, {method: 'POST', body: JSON.stringify(body)});

export const api = {
    getManifest: () => get('/assets/manifest.json'),
    createParticipant: body => post('/api/participants', body),
    getParticipant: id => get(`/api/participants/${id}`),
    submitTrial: body => post('/api/trials', body),
    submitAdjustment: body => post('/api/adjustments', body),
    finish: id => post(`/api/participants/${id}/finish`)
}