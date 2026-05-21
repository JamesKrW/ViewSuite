/**
 * UI controls — cluster selector, filters, clustering method, display toggles.
 */
class UIControls {
    constructor() {
        // Topbar cluster selector
        this.clusterDropdown = document.getElementById('cluster-dropdown');
        this.clusterSlider = document.getElementById('cluster-slider');
        this.compInfo = document.getElementById('comp-info');

        // Filters
        this.minDegreeInput = document.getElementById('filter-min-degree');
        this.applyBtn = document.getElementById('apply-filters');
        this.resetBtn = document.getElementById('reset-filters');

        // Clustering method
        this.clusterBySelect = document.getElementById('cluster-by');
        this.clusterGroupField = document.getElementById('cluster-group-field');
        this.clusterFieldGroup = document.getElementById('cluster-field-group');

        // Display
        this.showEdgesCheck = document.getElementById('show-edges');
        this.showLabelsCheck = document.getElementById('show-labels');
        this.showEdgeLabelsCheck = document.getElementById('show-edge-labels');
        this.hoverReachableCheck = document.getElementById('hover-reachable');
        this.hoverPassthroughCheck = document.getElementById('hover-passthrough');
        this.layoutXSelect = document.getElementById('layout-x');
        this.layoutYSelect = document.getElementById('layout-y');

        // Search
        this.searchInput = document.getElementById('search-input');
        this.searchResults = document.getElementById('search-results');

        // Callbacks
        this.onClusterSelect = null;       // (clusterIdx or null) — null = All Graph
        this.onClusterMethodChange = null;  // () — method or group field changed
        this.onFiltersApply = null;
        this.onShowEdgesChange = null;
        this.onShowLabelsChange = null;
        this.onShowEdgeLabelsChange = null;
        this.onHoverReachableChange = null;
        this.onHoverPassthroughChange = null;
        this.onLayoutChange = null;
        this.onNodeSelect = null;

        this._searchTimeout = null;
        this._sliderDebounce = null;
        this._bindEvents();
    }

    _bindEvents() {
        // Cluster dropdown (topbar)
        this.clusterDropdown.addEventListener('change', () => {
            const val = this.clusterDropdown.value;
            if (val === '') {
                this.onClusterSelect?.(null);  // All Graph
            } else {
                const idx = parseInt(val);
                this.clusterSlider.value = idx;
                this.onClusterSelect?.(idx);
            }
        });

        // Cluster slider
        this.clusterSlider.addEventListener('input', () => {
            const idx = parseInt(this.clusterSlider.value);
            this.clusterDropdown.value = String(idx);
            clearTimeout(this._sliderDebounce);
            this._sliderDebounce = setTimeout(() => {
                this.onClusterSelect?.(idx);
            }, 150);
        });

        // Clustering method (left panel)
        this.clusterBySelect.addEventListener('change', () => {
            const method = this.clusterBySelect.value;
            this.clusterFieldGroup.style.display = method === 'group' ? 'block' : 'none';
            this.onClusterMethodChange?.();
        });

        this.clusterGroupField.addEventListener('change', () => {
            if (this.clusterBySelect.value === 'group') {
                this.onClusterMethodChange?.();
            }
        });

        // Filters
        this.applyBtn.addEventListener('click', () => this.onFiltersApply?.(this.getFilters()));
        this.resetBtn.addEventListener('click', () => {
            this.minDegreeInput.value = '0';
            this.onFiltersApply?.(this.getFilters());
        });

        // Display toggles
        this.showEdgesCheck.addEventListener('change', () => {
            this.onShowEdgesChange?.(this.showEdgesCheck.checked);
        });
        this.showLabelsCheck.addEventListener('change', () => {
            this.onShowLabelsChange?.(this.showLabelsCheck.checked);
        });
        this.showEdgeLabelsCheck.addEventListener('change', () => {
            this.onShowEdgeLabelsChange?.(this.showEdgeLabelsCheck.checked);
        });
        this.hoverReachableCheck.addEventListener('change', () => {
            if (this.hoverReachableCheck.checked) this.hoverPassthroughCheck.checked = false;
            this.onHoverReachableChange?.(this.hoverReachableCheck.checked);
        });
        this.hoverPassthroughCheck.addEventListener('change', () => {
            if (this.hoverPassthroughCheck.checked) this.hoverReachableCheck.checked = false;
            this.onHoverPassthroughChange?.(this.hoverPassthroughCheck.checked);
        });

        // Layout
        this.layoutXSelect.addEventListener('change', () => {
            this.onLayoutChange?.(this.layoutXSelect.value, this.layoutYSelect.value);
        });
        this.layoutYSelect.addEventListener('change', () => {
            this.onLayoutChange?.(this.layoutXSelect.value, this.layoutYSelect.value);
        });

        // Search
        this.searchInput.addEventListener('input', () => {
            clearTimeout(this._searchTimeout);
            this._searchTimeout = setTimeout(() => this._doSearch(), 300);
        });
        this.searchInput.addEventListener('focus', () => {
            if (this.searchResults.children.length > 0)
                this.searchResults.style.display = 'block';
        });
        document.addEventListener('click', (e) => {
            if (!this.searchInput.contains(e.target) && !this.searchResults.contains(e.target))
                this.searchResults.style.display = 'none';
        });
        this.searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this._doSearch(); }
        });
    }

    // ── Populate methods ────────────────────────────────────────────

    populateClusters(clusters) {
        // clusters: [{index, label, node_count, edge_count?}, ...]
        this.clusterDropdown.innerHTML = '<option value="">All Graph</option>';
        for (const c of clusters) {
            const opt = document.createElement('option');
            opt.value = c.index;
            const ec = c.edge_count ? `, ${c.edge_count}e` : '';
            opt.textContent = `${c.label} (${c.node_count}n${ec})`;
            this.clusterDropdown.appendChild(opt);
        }
        if (clusters.length > 0) {
            this.clusterSlider.min = 0;
            this.clusterSlider.max = clusters.length - 1;
            this.clusterSlider.value = 0;
        }
    }

    populateClusterGroupFields(fields) {
        this.clusterGroupField.innerHTML = '';
        for (const f of fields) {
            const opt = document.createElement('option');
            opt.value = f.path;
            opt.textContent = f.path;
            this.clusterGroupField.appendChild(opt);
        }
    }

    populateLayoutFields(fields) {
        for (const sel of [this.layoutXSelect, this.layoutYSelect]) {
            sel.innerHTML = '<option value="">(auto)</option>';
            for (const f of fields) {
                const opt = document.createElement('option');
                opt.value = f.path;
                opt.textContent = `${f.path} [${f.min.toFixed(2)} .. ${f.max.toFixed(2)}]`;
                sel.appendChild(opt);
            }
        }
        const xCandidates = fields.filter(f => /\b(tx|x|lon)\b/i.test(f.path));
        const yCandidates = fields.filter(f => /\b(ty|y|lat)\b/i.test(f.path));
        if (xCandidates.length > 0) this.layoutXSelect.value = xCandidates[0].path;
        else if (fields.length > 0) this.layoutXSelect.value = fields[0].path;
        if (yCandidates.length > 0) this.layoutYSelect.value = yCandidates[0].path;
        else if (fields.length > 1) this.layoutYSelect.value = fields[1].path;
    }

    getLayoutFields() {
        return {
            x: this.layoutXSelect.value || null,
            y: this.layoutYSelect.value || null,
        };
    }

    getFilters() {
        const filters = {};
        const minDeg = parseInt(this.minDegreeInput.value) || 0;
        if (minDeg > 0) filters.minDegree = minDeg;
        return filters;
    }

    getClusterMethod() {
        return this.clusterBySelect.value;
    }

    getClusterField() {
        return this.clusterBySelect.value === 'group' ? this.clusterGroupField.value : null;
    }

    setClusterDropdown(idx) {
        this.clusterDropdown.value = idx !== null ? String(idx) : '';
        if (idx !== null) this.clusterSlider.value = idx;
    }

    setCompInfo(text) {
        this.compInfo.textContent = text;
    }

    // ── Search ──────────────────────────────────────────────────────

    async _doSearch() {
        const q = this.searchInput.value.trim();
        if (q.length < 2) { this.searchResults.style.display = 'none'; return; }

        const data = await API.search(q);
        this.searchResults.innerHTML = '';

        if (data.nodes.length === 0) {
            this.searchResults.innerHTML = '<div class="result-item"><span class="muted">No results</span></div>';
        } else {
            for (const node of data.nodes) {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `<span class="result-id">${node.id}</span>
                    <span class="result-scene">comp #${node.component}</span>`;
                item.addEventListener('click', () => {
                    this.searchResults.style.display = 'none';
                    this.searchInput.value = '';
                    this.onNodeSelect?.(node.id, node.component);
                });
                this.searchResults.appendChild(item);
            }
        }
        this.searchResults.style.display = 'block';
    }
}
