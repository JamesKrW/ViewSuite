/**
 * Node inspector — shows all node attributes on click.
 */
class Inspector {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
    }

    async showNode(nodeId) {
        this.container.innerHTML = '<p class="muted">Loading...</p>';
        try {
            const data = await API.getNode(nodeId);
            this._renderNode(data);
        } catch (err) {
            this.container.innerHTML = `<p class="muted">Error: ${err.message}</p>`;
        }
    }

    _renderNode(data) {
        let html = '';

        // Node ID (prominent)
        html += this._field('Node ID', data.id);

        // Component
        html += this._field('Component', data.component);

        // Degree
        html += this._field('Degree', `In: ${data.in_degree} / Out: ${data.out_degree}`);

        // Observation string
        if (data.obs_str) {
            html += this._field('obs_str', data.obs_str);
        }

        // State — render all fields recursively
        if (data.state && typeof data.state === 'object' && Object.keys(data.state).length > 0) {
            html += '<div class="inspector-field"><div class="field-label">STATE</div>';
            html += this._renderObj(data.state, 1);
            html += '</div>';
        }

        // Extra
        if (data.extra && typeof data.extra === 'object' && Object.keys(data.extra).length > 0) {
            html += '<div class="inspector-field"><div class="field-label">EXTRA</div>';
            html += this._renderObj(data.extra, 1);
            html += '</div>';
        }

        // Images
        if (data.image_paths && data.image_paths.length > 0) {
            html += '<div class="inspector-field"><div class="field-label">IMAGES</div></div>';
            for (const imgPath of data.image_paths) {
                const url = API.imageUrl(imgPath);
                html += `<img class="inspector-image" src="${url}"
                    onclick="showLightbox('${url}')" loading="lazy" alt="Node image" />`;
            }
        }

        // Outgoing edges
        if (data.out_edges && data.out_edges.length > 0) {
            html += `<div class="inspector-field"><div class="field-label">OUTGOING EDGES (${data.out_edges.length})</div></div>`;
            html += '<div class="edge-list">';
            for (const e of data.out_edges) {
                const shortTo = e.to.length > 20 ? '...' + e.to.slice(-12) : e.to;
                html += `<div class="edge-item" onclick="app.navigateToNode('${this._esc(e.to)}')">
                    <span class="edge-arrow">&rarr;</span>
                    <span class="edge-action">${this._escapeHtml(e.action)}</span>
                    <span class="edge-node">${this._escapeHtml(shortTo)}</span>
                </div>`;
            }
            html += '</div>';
        }

        // Incoming edges
        if (data.in_edges && data.in_edges.length > 0) {
            html += `<div class="inspector-field"><div class="field-label">INCOMING EDGES (${data.in_edges.length})</div></div>`;
            html += '<div class="edge-list">';
            for (const e of data.in_edges) {
                const shortFrom = e.from.length > 20 ? '...' + e.from.slice(-12) : e.from;
                html += `<div class="edge-item" onclick="app.navigateToNode('${this._esc(e.from)}')">
                    <span class="edge-arrow">&larr;</span>
                    <span class="edge-action">${this._escapeHtml(e.action)}</span>
                    <span class="edge-node">${this._escapeHtml(shortFrom)}</span>
                </div>`;
            }
            html += '</div>';
        }

        this.container.innerHTML = html;
    }

    _renderObj(obj, depth) {
        let html = '<div style="margin-left:' + (depth * 10) + 'px">';
        for (const [key, val] of Object.entries(obj)) {
            if (val && typeof val === 'object' && !Array.isArray(val)) {
                html += `<div class="field-label" style="margin-top:4px">${this._escapeHtml(key)}</div>`;
                html += this._renderObj(val, depth + 1);
            } else {
                const display = typeof val === 'number'
                    ? (Number.isInteger(val) ? val : val.toFixed(4))
                    : JSON.stringify(val);
                html += `<div class="inspector-kv">
                    <span class="kv-key">${this._escapeHtml(key)}</span>
                    <span class="kv-val">${this._escapeHtml(String(display))}</span>
                </div>`;
            }
        }
        html += '</div>';
        return html;
    }

    _field(label, value) {
        return `<div class="inspector-field">
            <div class="field-label">${this._escapeHtml(label)}</div>
            <div class="field-value">${this._escapeHtml(String(value))}</div>
        </div>`;
    }

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    _esc(str) {
        return str.replace(/'/g, "\\'");
    }

    clear() {
        this.container.innerHTML = '<p class="muted">Click a node to inspect its attributes</p>';
    }
}

function showLightbox(url) {
    document.getElementById('lightbox-img').src = url;
    document.getElementById('lightbox').style.display = 'flex';
}

function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
}

function openModal(title, bodyHtml) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = bodyHtml;
    document.getElementById('modal-overlay').style.display = 'flex';
}

function closeModal() {
    document.getElementById('modal-overlay').style.display = 'none';
    document.querySelectorAll('#modal-body canvas').forEach(c => {
        const chart = Chart.getChart(c);
        if (chart) chart.destroy();
    });
}
