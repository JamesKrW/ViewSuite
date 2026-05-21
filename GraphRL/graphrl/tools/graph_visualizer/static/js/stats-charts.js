/**
 * Statistics charts — pure graph-theoretic metrics.
 */
Chart.defaults.color = '#9ba1b0';
Chart.defaults.borderColor = '#2d3240';

class StatsCharts {
    static async showComponentCharts(compIdx) {
        const stats = await API.getComponentStats(compIdx);

        let html = `<div class="chart-container"><h4>Component #${compIdx} Summary</h4>
            <table class="scenes-table">
            <tr><td>Nodes</td><td>${stats.node_count}</td></tr>
            <tr><td>Edges</td><td>${stats.edge_count}</td></tr>
            <tr><td>Avg Degree</td><td>${stats.avg_degree}</td></tr>
            <tr><td>Max Degree</td><td>${stats.max_degree}</td></tr>
            <tr><td>Max In-Degree</td><td>${stats.max_in_degree}</td></tr>
            <tr><td>Max Out-Degree</td><td>${stats.max_out_degree}</td></tr>
            <tr><td>Source Nodes (in=0)</td><td>${stats.source_nodes}</td></tr>
            <tr><td>Sink Nodes (out=0)</td><td>${stats.sink_nodes}</td></tr>
            ${stats.diameter >= 0 ? `<tr><td>Diameter</td><td>${stats.diameter}</td></tr>` : ''}
            <tr><td>Strongly Connected Components</td><td>${stats.strongly_connected_components}</td></tr>
            <tr><td>Non-trivial SCCs (size>1)</td><td>${stats.nontrivial_scc_count}</td></tr>
            <tr><td>Largest SCC Size</td><td>${stats.largest_scc_size}</td></tr>
            </table></div>`;

        html += '<div class="chart-row">';
        html += '<div class="chart-container chart-half"><h4>Degree Distribution</h4>';
        html += '<canvas id="chart-degree"></canvas></div>';

        html += '<div class="chart-container chart-half"><h4>In-Degree vs Out-Degree</h4>';
        html += '<canvas id="chart-in-out"></canvas></div>';
        html += '</div>';

        if (stats.scc_sizes && stats.scc_sizes.length > 1) {
            html += '<div class="chart-container"><h4>Strongly Connected Component Sizes</h4>';
            html += '<canvas id="chart-scc"></canvas></div>';
        }

        openModal(`Component #${compIdx} — Graph Stats`, html);

        requestAnimationFrame(() => {
            // Degree distribution
            new Chart(document.getElementById('chart-degree'), {
                type: 'bar',
                data: {
                    labels: Object.keys(stats.degree_distribution),
                    datasets: [{ label: 'Count', data: Object.values(stats.degree_distribution),
                        backgroundColor: 'rgba(99,102,241,0.6)', borderColor: 'rgba(99,102,241,1)', borderWidth: 1 }]
                },
                options: { responsive: true, plugins: { legend: { display: false } },
                    scales: { x: { title: { display: true, text: 'Degree' } },
                              y: { title: { display: true, text: 'Count' }, beginAtZero: true } } }
            });

            // In vs Out degree
            const allDegs = new Set([
                ...Object.keys(stats.in_degree_distribution),
                ...Object.keys(stats.out_degree_distribution)
            ]);
            const degLabels = [...allDegs].sort((a, b) => parseInt(a) - parseInt(b));
            new Chart(document.getElementById('chart-in-out'), {
                type: 'bar',
                data: {
                    labels: degLabels,
                    datasets: [
                        { label: 'In-Degree', data: degLabels.map(d => stats.in_degree_distribution[d] || 0),
                          backgroundColor: 'rgba(59,130,246,0.6)', borderColor: 'rgba(59,130,246,1)', borderWidth: 1 },
                        { label: 'Out-Degree', data: degLabels.map(d => stats.out_degree_distribution[d] || 0),
                          backgroundColor: 'rgba(239,68,68,0.6)', borderColor: 'rgba(239,68,68,1)', borderWidth: 1 },
                    ]
                },
                options: { responsive: true,
                    scales: { x: { title: { display: true, text: 'Degree' } },
                              y: { beginAtZero: true } } }
            });

            // SCC sizes
            const sccCanvas = document.getElementById('chart-scc');
            if (sccCanvas && stats.scc_sizes.length > 1) {
                new Chart(sccCanvas, {
                    type: 'bar',
                    data: {
                        labels: stats.scc_sizes.map((_, i) => `SCC ${i}`),
                        datasets: [{ label: 'Nodes', data: stats.scc_sizes,
                            backgroundColor: stats.scc_sizes.map((_, i) =>
                                COMPONENT_COLORS[i % COMPONENT_COLORS.length] + 'aa'), borderWidth: 0 }]
                    },
                    options: { responsive: true, plugins: { legend: { display: false } },
                        scales: { y: { beginAtZero: true, title: { display: true, text: 'Nodes' } } } }
                });
            }
        });
    }

    static async showComponentsTable() {
        const data = await API.getComponentsTable();
        const comps = data.components;

        let html = '<div style="max-height:60vh;overflow-y:auto;">';
        html += `<table class="scenes-table" id="comp-table-inner">
            <thead><tr>
                <th data-col="index"># <span class="sort-arrow"></span></th>
                <th data-col="node_count">Nodes <span class="sort-arrow"></span></th>
                <th data-col="edge_count">Edges <span class="sort-arrow"></span></th>
                <th data-col="avg_degree">Avg Degree <span class="sort-arrow"></span></th>
            </tr></thead>
            <tbody id="comp-table-body"></tbody>
        </table></div>`;

        openModal(`All Connected Components (${comps.length})`, html);

        const tbody = document.getElementById('comp-table-body');
        let sortState = { col: 'node_count', asc: false };

        function renderRows(sorted) {
            tbody.innerHTML = '';
            for (const c of sorted) {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${c.index}</td><td>${c.node_count}</td><td>${c.edge_count}</td><td>${c.avg_degree}</td>`;
                tr.addEventListener('click', () => { closeModal(); app.loadComponent(c.index); });
                tbody.appendChild(tr);
            }
        }

        function sortAndRender() {
            const sorted = [...comps].sort((a, b) =>
                sortState.asc ? a[sortState.col] - b[sortState.col] : b[sortState.col] - a[sortState.col]);
            renderRows(sorted);
        }

        document.querySelectorAll('#comp-table-inner th').forEach(th => {
            th.addEventListener('click', () => {
                const col = th.dataset.col;
                sortState = { col, asc: sortState.col === col ? !sortState.asc : false };
                sortAndRender();
            });
        });

        sortAndRender();
    }

    static async showGlobalCharts() {
        const stats = await API.getGlobalStats();

        let html = `<div class="chart-container"><h4>Global Graph Summary</h4>
            <table class="scenes-table">
            <tr><td>Total Nodes</td><td>${stats.total_nodes.toLocaleString()}</td></tr>
            <tr><td>Total Edges</td><td>${stats.total_edges.toLocaleString()}</td></tr>
            <tr><td>Connected Components (weakly)</td><td>${stats.total_components.toLocaleString()}</td></tr>
            <tr><td>Strongly Connected Components</td><td>${stats.total_strongly_connected_components.toLocaleString()}</td></tr>
            <tr><td>Non-trivial SCCs (size>1)</td><td>${stats.nontrivial_scc_count}</td></tr>
            <tr><td>Largest SCC Size</td><td>${stats.largest_scc_size}</td></tr>
            <tr><td>Avg Degree</td><td>${stats.avg_degree}</td></tr>
            <tr><td>Max Degree</td><td>${stats.max_degree}</td></tr>
            <tr><td>Max In-Degree</td><td>${stats.max_in_degree}</td></tr>
            <tr><td>Max Out-Degree</td><td>${stats.max_out_degree}</td></tr>
            <tr><td>Isolated Nodes (degree=0)</td><td>${stats.isolated_nodes}</td></tr>
            <tr><td>Leaf Nodes (degree=1)</td><td>${stats.leaf_nodes}</td></tr>
            <tr><td>Source Nodes (in=0, out>0)</td><td>${stats.source_nodes}</td></tr>
            <tr><td>Sink Nodes (out=0, in>0)</td><td>${stats.sink_nodes}</td></tr>
            </table></div>`;

        html += '<div class="chart-container"><h4>Connected Component Size Distribution (top 100)</h4>';
        html += '<canvas id="chart-comp-sizes"></canvas></div>';

        html += '<div class="chart-row">';
        html += '<div class="chart-container chart-half"><h4>Degree Distribution</h4>';
        html += '<canvas id="chart-global-degree"></canvas></div>';

        html += '<div class="chart-container chart-half"><h4>In-Degree vs Out-Degree</h4>';
        html += '<canvas id="chart-global-in-out"></canvas></div>';
        html += '</div>';

        if (stats.scc_sizes && stats.scc_sizes.length > 1) {
            html += '<div class="chart-container"><h4>Strongly Connected Component Sizes (top 100)</h4>';
            html += '<canvas id="chart-global-scc"></canvas></div>';
        }

        openModal('Global Graph Statistics', html);

        requestAnimationFrame(() => {
            // Component sizes
            new Chart(document.getElementById('chart-comp-sizes'), {
                type: 'bar',
                data: {
                    labels: stats.component_sizes.map((_, i) => `#${i}`),
                    datasets: [{ label: 'Nodes', data: stats.component_sizes,
                        backgroundColor: stats.component_sizes.map((_, i) =>
                            COMPONENT_COLORS[i % COMPONENT_COLORS.length] + 'aa'), borderWidth: 0 }]
                },
                options: { responsive: true, plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, title: { display: true, text: 'Nodes' } } } }
            });

            // Degree distribution
            new Chart(document.getElementById('chart-global-degree'), {
                type: 'bar',
                data: {
                    labels: Object.keys(stats.degree_distribution),
                    datasets: [{ label: 'Count', data: Object.values(stats.degree_distribution),
                        backgroundColor: 'rgba(99,102,241,0.6)', borderColor: 'rgba(99,102,241,1)', borderWidth: 1 }]
                },
                options: { responsive: true, plugins: { legend: { display: false } },
                    scales: { x: { title: { display: true, text: 'Degree' } }, y: { beginAtZero: true } } }
            });

            // In vs Out
            const allDegs = new Set([
                ...Object.keys(stats.in_degree_distribution),
                ...Object.keys(stats.out_degree_distribution)
            ]);
            const degLabels = [...allDegs].sort((a, b) => parseInt(a) - parseInt(b));
            new Chart(document.getElementById('chart-global-in-out'), {
                type: 'bar',
                data: {
                    labels: degLabels,
                    datasets: [
                        { label: 'In-Degree', data: degLabels.map(d => stats.in_degree_distribution[d] || 0),
                          backgroundColor: 'rgba(59,130,246,0.6)', borderColor: 'rgba(59,130,246,1)', borderWidth: 1 },
                        { label: 'Out-Degree', data: degLabels.map(d => stats.out_degree_distribution[d] || 0),
                          backgroundColor: 'rgba(239,68,68,0.6)', borderColor: 'rgba(239,68,68,1)', borderWidth: 1 },
                    ]
                },
                options: { responsive: true,
                    scales: { x: { title: { display: true, text: 'Degree' } }, y: { beginAtZero: true } } }
            });

            // SCC sizes
            const sccCanvas = document.getElementById('chart-global-scc');
            if (sccCanvas) {
                new Chart(sccCanvas, {
                    type: 'bar',
                    data: {
                        labels: stats.scc_sizes.map((_, i) => `SCC ${i}`),
                        datasets: [{ label: 'Nodes', data: stats.scc_sizes,
                            backgroundColor: 'rgba(168,85,247,0.6)', borderColor: 'rgba(168,85,247,1)', borderWidth: 1 }]
                    },
                    options: { responsive: true, plugins: { legend: { display: false } },
                        scales: { y: { beginAtZero: true, title: { display: true, text: 'Nodes' } } } }
                });
            }
        });
    }
}
