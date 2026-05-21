/**
 * API client — generic clustering support.
 */
const API = {
    async getOverview(limit = 5000) {
        const res = await fetch(`/api/overview?limit=${limit}`);
        return res.json();
    },

    async getClusters(method = 'wcc', field = null) {
        const params = new URLSearchParams({ method });
        if (field) params.set('field', field);
        const res = await fetch(`/api/clusters?${params}`);
        return res.json();
    },

    async getClusterGraph(method, idx, field = null, filters = {}) {
        const params = new URLSearchParams({ method, idx });
        if (field) params.set('field', field);
        if (filters.minDegree) params.set('min_degree', filters.minDegree);
        const res = await fetch(`/api/cluster_graph?${params}`);
        return res.json();
    },

    async getNode(nodeId) {
        const res = await fetch(`/api/node/${encodeURIComponent(nodeId)}`);
        return res.json();
    },

    async getGlobalStats() {
        const res = await fetch('/api/stats/global');
        return res.json();
    },

    async getComponentStats(compIdx) {
        const res = await fetch(`/api/stats/component/${compIdx}`);
        return res.json();
    },

    async getComponentsTable() {
        const res = await fetch('/api/stats/components_table');
        return res.json();
    },

    async findPath(fromId, toId) {
        const res = await fetch(`/api/path/${encodeURIComponent(fromId)}/${encodeURIComponent(toId)}`);
        return res.json();
    },

    async search(query) {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        return res.json();
    },

    async getLayoutFields() {
        const res = await fetch('/api/layout_fields');
        return res.json();
    },

    async getGroupFields() {
        const res = await fetch('/api/group_fields');
        return res.json();
    },

    imageUrl(imagePath) {
        return `/api/images/${imagePath.replace('images/', '')}`;
    }
};
