/**
 * Main application controller.
 * Default: overview (one dot per cluster). Click/select to enter cluster subgraph.
 */
class App {
    constructor() {
        this.renderer = new GraphRenderer('sigma-container');
        this.inspector = new Inspector('inspector-content');
        this.controls = new UIControls();
        this.isOverview = true;
        this.currentClusterIdx = null;
        this.currentViewData = null;

        this._bindEvents();
        this._init();
    }

    async _init() {
        this.setStatus('Loading...');

        const [fieldsData, groupFieldsData] = await Promise.all([
            API.getLayoutFields(),
            API.getGroupFields().catch(() => ({ fields: [] })),
        ]);

        this.controls.populateLayoutFields(fieldsData.fields);
        this.controls.populateClusterGroupFields(groupFieldsData.fields || []);

        const layout = this.controls.getLayoutFields();
        this.renderer.layoutX = layout.x;
        this.renderer.layoutY = layout.y;

        // Load default clustering and overview
        await this._refreshClusters();
    }

    _bindEvents() {
        // Cluster selection (topbar dropdown/slider)
        this.controls.onClusterSelect = (idx) => {
            if (idx === null) {
                this._loadOverview();
            } else {
                this._loadCluster(idx);
            }
        };

        // Clustering method change (left panel)
        this.controls.onClusterMethodChange = () => this._refreshClusters();

        // Filters
        this.controls.onFiltersApply = (filters) => {
            if (!this.isOverview && this.currentClusterIdx !== null) {
                this._loadCluster(this.currentClusterIdx, filters);
            }
        };

        // Display toggles
        this.controls.onShowEdgesChange = (show) => this.renderer.setShowEdges(show);
        this.controls.onShowLabelsChange = (show) => this.renderer.setShowLabels(show);
        this.controls.onShowEdgeLabelsChange = (show) => this.renderer.setShowEdgeLabels(show);
        this.controls.onHoverReachableChange = (show) => this.renderer.setHoverShowReachable(show);
        this.controls.onHoverPassthroughChange = (show) => this.renderer.setHoverShowPassthrough(show);

        this.controls.onLayoutChange = (x, y) => {
            this.renderer.layoutX = x;
            this.renderer.layoutY = y;
            if (this.currentViewData) {
                this.renderer.loadData(this.currentViewData);
            }
        };

        // Node click — always open inspector
        this.renderer.onNodeClick = (nodeId) => {
            this.inspector.showNode(nodeId);
        };

        // Search → navigate to node
        this.controls.onNodeSelect = (nodeId, compIdx) => {
            // Switch to WCC mode and load that component
            this.controls.clusterBySelect.value = 'wcc';
            this._refreshClusters().then(() => {
                this.controls.setClusterDropdown(compIdx);
                this._loadCluster(compIdx).then(() => {
                    this.renderer.focusNode(nodeId);
                    this.inspector.showNode(nodeId);
                });
            });
        };

        // Stats buttons
        document.getElementById('show-charts-btn').addEventListener('click', () => {
            if (this.currentClusterIdx !== null) {
                StatsCharts.showComponentCharts(this.currentClusterIdx);
            }
        });
        document.getElementById('show-comp-table-btn').addEventListener('click', () => {
            StatsCharts.showComponentsTable();
        });
        document.getElementById('show-global-stats-btn').addEventListener('click', () => {
            StatsCharts.showGlobalCharts();
        });

        document.getElementById('find-path-btn').addEventListener('click', () => this._findPath());

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
                closeLightbox();
                this.renderer.clearPathHighlight();
            }
        });
    }

    // ── Core flow ───────────────────────────────────────────────────

    async _refreshClusters() {
        const method = this.controls.getClusterMethod();
        const field = this.controls.getClusterField();

        this.setStatus('Loading clusters...');
        const data = await API.getClusters(method, field);
        this.controls.populateClusters(data.clusters);

        document.getElementById('global-info').textContent =
            `${data.total} clusters (${method})`;

        // Show overview
        await this._loadOverview();
    }

    async _loadOverview() {
        this.setStatus('Loading overview (top nodes by degree)...');
        document.getElementById('loading-overlay').style.display = 'flex';
        document.getElementById('graph-placeholder').style.display = 'none';

        try {
            const data = await API.getOverview(5000);
            this.currentViewData = data;
            this.isOverview = true;
            this.currentClusterIdx = null;
            this.controls.setClusterDropdown(null);

            this.renderer.loadData(data);
            this.renderer.setColorMode('degree');

            this.controls.setCompInfo(`${data.nodes.length}n / ${data.edges.length}e`);
            this._updateOverviewStats(data);
            this.setStatus(`All Graph: showing top ${data.nodes.length} nodes by degree, ${data.edges.length} edges`);
            document.getElementById('show-charts-btn').disabled = true;
        } catch (err) {
            this.setStatus(`Error: ${err.message}`);
        } finally {
            document.getElementById('loading-overlay').style.display = 'none';
        }
    }

    async _loadCluster(clusterIdx, filters = {}) {
        const method = this.controls.getClusterMethod();
        const field = this.controls.getClusterField();

        this.setStatus(`Loading cluster #${clusterIdx}...`);
        document.getElementById('loading-overlay').style.display = 'flex';
        document.getElementById('graph-placeholder').style.display = 'none';

        try {
            const data = await API.getClusterGraph(method, clusterIdx, field, filters);
            this.currentViewData = data;
            this.isOverview = false;
            this.currentClusterIdx = clusterIdx;

            this.renderer.loadData(data);
            this.renderer.setColorMode('degree');

            this.controls.setCompInfo(`${data.nodes.length}n / ${data.edges.length}e`);
            this._updateClusterStats(clusterIdx, data);
            document.getElementById('show-charts-btn').disabled = false;
            this.inspector.clear();

            this.setStatus(`Cluster #${clusterIdx}: ${data.nodes.length} nodes, ${data.edges.length} edges`);
        } catch (err) {
            this.setStatus(`Error: ${err.message}`);
        } finally {
            document.getElementById('loading-overlay').style.display = 'none';
        }
    }

    // ── Stats ───────────────────────────────────────────────────────

    _updateOverviewStats(data) {
        const container = document.getElementById('comp-stats-summary');
        const nodes = data.nodes;
        const totalDegree = nodes.reduce((s, n) => s + n.degree, 0);
        const avgDegree = nodes.length > 0 ? (totalDegree / nodes.length).toFixed(2) : '0';
        const maxDegree = nodes.length > 0 ? Math.max(...nodes.map(n => n.degree)) : 0;
        container.innerHTML = `
            <div class="stat-row"><span class="stat-label">View</span><span class="stat-value">All Graph (LOD)</span></div>
            <div class="stat-row"><span class="stat-label">Shown Nodes</span><span class="stat-value">${nodes.length.toLocaleString()}</span></div>
            <div class="stat-row"><span class="stat-label">Shown Edges</span><span class="stat-value">${data.edges.length.toLocaleString()}</span></div>
            <div class="stat-row"><span class="stat-label">Avg Degree</span><span class="stat-value">${avgDegree}</span></div>
            <div class="stat-row"><span class="stat-label">Max Degree</span><span class="stat-value">${maxDegree}</span></div>
            <div class="stat-row"><span class="stat-label">WCC</span><span class="stat-value">${data.num_sub_components || '-'}</span></div>
        `;
    }

    _updateClusterStats(clusterIdx, data) {
        const container = document.getElementById('comp-stats-summary');
        const nodes = data.nodes;
        const totalDegree = nodes.reduce((s, n) => s + n.degree, 0);
        const avgDegree = nodes.length > 0 ? (totalDegree / nodes.length).toFixed(2) : '0';
        const maxDegree = nodes.length > 0 ? Math.max(...nodes.map(n => n.degree)) : 0;

        container.innerHTML = `
            <div class="stat-row"><span class="stat-label">Cluster</span><span class="stat-value">#${clusterIdx}</span></div>
            <div class="stat-row"><span class="stat-label">Nodes</span><span class="stat-value">${nodes.length.toLocaleString()}</span></div>
            <div class="stat-row"><span class="stat-label">Edges</span><span class="stat-value">${data.edges.length.toLocaleString()}</span></div>
            <div class="stat-row"><span class="stat-label">Avg Degree</span><span class="stat-value">${avgDegree}</span></div>
            <div class="stat-row"><span class="stat-label">Max Degree</span><span class="stat-value">${maxDegree}</span></div>
            <div class="stat-row"><span class="stat-label">WCC</span><span class="stat-value">${data.num_sub_components || '-'}</span></div>
            <div class="stat-row"><span class="stat-label">SCC</span><span class="stat-value">${data.num_sccs || '-'}</span></div>
        `;
    }

    // ── Path finder ─────────────────────────────────────────────────

    async _findPath() {
        const fromId = document.getElementById('path-from').value.trim();
        const toId = document.getElementById('path-to').value.trim();
        const resultDiv = document.getElementById('path-result');

        if (!fromId || !toId) {
            resultDiv.innerHTML = '<p class="muted">Enter both node IDs</p>';
            return;
        }

        resultDiv.innerHTML = '<p class="muted">Searching...</p>';

        try {
            const data = await API.findPath(fromId, toId);
            if (data.error) {
                resultDiv.innerHTML = `<p class="muted">${data.error}</p>`;
                this.renderer.clearPathHighlight();
                return;
            }
            if (data.path.length === 0) {
                resultDiv.innerHTML = '<p class="muted">No path found</p>';
                this.renderer.clearPathHighlight();
                return;
            }

            this.renderer.highlightPath(data.path, data.edges);

            let html = `<p><strong>Path length: ${data.distance}</strong></p>`;
            for (let i = 0; i < data.path.length; i++) {
                const nid = data.path[i];
                const short = nid.length > 16 ? '...' + nid.slice(-12) : nid;
                html += `<div class="path-step">
                    <span onclick="app.navigateToNode('${nid.replace(/'/g, "\\'")}')" style="cursor:pointer;color:var(--accent)">${short}</span>`;
                if (i < data.edges.length) {
                    html += ` <span class="path-action">&rarr; ${data.edges[i].action}</span>`;
                }
                html += '</div>';
            }
            html += `<button class="btn btn-secondary mt-8" onclick="app.renderer.clearPathHighlight()">Clear Path</button>`;
            resultDiv.innerHTML = html;
        } catch (err) {
            resultDiv.innerHTML = `<p class="muted">Error: ${err.message}</p>`;
        }
    }

    async navigateToNode(nodeId) {
        // If the node is already in the current graph, just highlight it
        if (this.renderer.graph && this.renderer.graph.hasNode(nodeId)) {
            this.renderer.focusNode(nodeId);
            return;
        }

        // Node not in current view — load its WCC component, then highlight
        try {
            const nodeData = await API.getNode(nodeId);
            if (nodeData.error) {
                this.setStatus(`Node not found: ${nodeId}`);
                return;
            }
            const compIdx = nodeData.component;
            if (compIdx === undefined || compIdx < 0) {
                this.setStatus(`Cannot locate component for node`);
                return;
            }

            this.controls.clusterBySelect.value = 'wcc';
            const clusterData = await API.getClusters('wcc');
            this.controls.populateClusters(clusterData.clusters);
            document.getElementById('global-info').textContent =
                `${clusterData.total} clusters (wcc)`;

            this.controls.setClusterDropdown(compIdx);
            await this._loadCluster(compIdx);
            this.renderer.focusNode(nodeId);
        } catch (err) {
            this.setStatus(`Error navigating: ${err.message}`);
        }
    }

    setStatus(text) {
        document.getElementById('status-text').textContent = text;
    }
}

const app = new App();
