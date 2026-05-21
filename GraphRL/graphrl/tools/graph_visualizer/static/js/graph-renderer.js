/**
 * Sigma.js graph rendering — generic, layout by user-selected fields.
 */

// Edge action colors (common actions get distinct colors)
const ACTION_COLORS = {
    'move_forward':  '#3b82f6',
    'move_backward': '#ef4444',
    'move_left':     '#22c55e',
    'move_right':    '#f97316',
    'move_up':       '#a855f7',
    'move_down':     '#92400e',
    'turn_left':     '#06b6d4',
    'turn_right':    '#ec4899',
    'look_up':       '#84cc16',
    'look_down':     '#f43f5e',
};

// Sub-component colors (20 distinct)
const COMPONENT_COLORS = [
    '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
    '#a855f7', '#f97316', '#ec4899', '#84cc16', '#14b8a6',
    '#8b5cf6', '#f43f5e', '#0ea5e9', '#d946ef', '#eab308',
    '#10b981', '#e11d48', '#7c3aed', '#059669', '#dc2626',
];

const DEGREE_GRADIENT = [
    '#1e3a5f', '#1e5490', '#2563eb', '#3b82f6', '#60a5fa',
    '#93c5fd', '#f59e0b', '#f97316', '#ef4444', '#dc2626',
];

function getActionColor(actionStr) {
    const parts = actionStr.split('|').map(s => s.trim());
    for (const p of parts) {
        if (ACTION_COLORS[p]) return ACTION_COLORS[p];
    }
    // Hash-based fallback for unknown actions
    let hash = 0;
    for (let i = 0; i < actionStr.length; i++) {
        hash = actionStr.charCodeAt(i) + ((hash << 5) - hash);
    }
    return COMPONENT_COLORS[Math.abs(hash) % COMPONENT_COLORS.length];
}

function getDegreeColor(degree, maxDegree) {
    const ratio = Math.min(degree / Math.max(maxDegree, 1), 1);
    const idx = Math.floor(ratio * (DEGREE_GRADIENT.length - 1));
    return DEGREE_GRADIENT[idx];
}

class GraphRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.graph = null;
        this.renderer = null;
        this.hoveredNode = null;
        this.selectedNode = null;
        this.highlightedPath = new Set();
        this.highlightedPathEdges = new Set();
        this.reachableNodes = new Set();   // BFS reachable from hovered/selected
        this.reachableEdges = new Set();   // edges on the reachable tree
        this.onNodeClick = null;
        this.colorMode = 'degree';
        this.showEdges = true;
        this.showLabels = false;
        this.showEdgeLabels = false;
        this.hoverShowReachable = false;   // toggle: show all reachable vs direct neighbors
        this.hoverShowPassthrough = false; // toggle: show all nodes on paths through this node
        this.currentData = null;
        this.layoutX = null;  // field path for X axis
        this.layoutY = null;  // field path for Y axis
    }

    loadData(data) {
        this.currentData = data;
        this.highlightedPath.clear();
        this.highlightedPathEdges.clear();
        this.selectedNode = null;

        if (this.renderer) {
            this.renderer.kill();
            this.renderer = null;
        }

        this.graph = new graphology.Graph({ multi: true, type: 'directed' });

        if (!data.nodes || data.nodes.length === 0) return;

        const maxDegree = Math.max(1, ...data.nodes.map(n => n.degree));

        // Determine node positions and detect overlaps
        const posMap = new Map();  // "x,y" → [nodeData...]
        const nodePositions = [];
        for (const n of data.nodes) {
            let x = 0, y = 0;
            if (n.x !== undefined) {
                x = n.x;
            } else if (this.layoutX && n[this.layoutX] !== undefined) {
                x = n[this.layoutX];
            } else {
                x = this._hashPos(n.id, 0);
            }
            if (n.y !== undefined) {
                y = n.y;
            } else if (this.layoutY && n[this.layoutY] !== undefined) {
                y = -n[this.layoutY];
            } else {
                y = -this._hashPos(n.id, 1);
            }
            // Round to detect near-overlaps (within 0.001)
            const key = `${Math.round(x * 1000)},${Math.round(y * 1000)}`;
            if (!posMap.has(key)) posMap.set(key, []);
            const group = posMap.get(key);
            group.push(n);
            nodePositions.push({ n, x, y, key, indexInGroup: group.length - 1 });
        }

        // Compute spread radius based on data range
        let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
        for (const p of nodePositions) {
            if (p.x < xMin) xMin = p.x;
            if (p.x > xMax) xMax = p.x;
            if (p.y < yMin) yMin = p.y;
            if (p.y > yMax) yMax = p.y;
        }
        const range = Math.max(xMax - xMin, yMax - yMin, 0.01);
        const jitterRadius = range * 0.008;

        for (const p of nodePositions) {
            let { x, y } = p;
            const groupSize = posMap.get(p.key).length;
            if (groupSize > 1) {
                // Spread overlapping nodes in a circle
                const angle = (2 * Math.PI * p.indexInGroup) / groupSize;
                const r = jitterRadius * Math.sqrt(groupSize);
                x += Math.cos(angle) * r;
                y += Math.sin(angle) * r;
            }

            const n = p.n;
            const color = this.colorMode === 'component'
                ? COMPONENT_COLORS[n.sub_component % COMPONENT_COLORS.length]
                : getDegreeColor(n.degree, maxDegree);

            this.graph.addNode(n.id, {
                x, y,
                size: Math.log2(n.degree + 2) * 2.5 + 2,
                color,
                label: n.label || n.id,
                degree: n.degree,
                sub_component: n.sub_component,
                originalColor: color,
            });
        }

        // Add edges — detect bidirectional pairs, use curved type for reverse edges
        const hasCurve = !!(window.EdgeCurvedArrowProgram || window.EdgeCurveProgram);
        const edgePairSet = new Set();
        for (const e of data.edges) {
            if (!this.graph.hasNode(e.source) || !this.graph.hasNode(e.target)) continue;
            const pairKey = `${e.source}→${e.target}`;
            const reverseKey = `${e.target}→${e.source}`;
            const isReverse = edgePairSet.has(reverseKey);
            edgePairSet.add(pairKey);

            try {
                const attrs = {
                    color: getActionColor(e.action),
                    size: 0.5,
                    action: e.action,
                    label: e.action,
                    originalColor: getActionColor(e.action),
                    id: e.id,
                };
                // Curve the reverse edge of a bidirectional pair
                if (isReverse && hasCurve) {
                    attrs.type = 'curved';
                    attrs.curvature = 0.25;
                }
                this.graph.addEdge(e.source, e.target, attrs);
            } catch (_) {}
        }

        // Build Sigma settings
        const sigmaSettings = {
            renderEdgeLabels: this.showEdgeLabels,
            enableEdgeClickEvents: true,
            enableEdgeHoverEvents: false,
            defaultEdgeType: 'arrow',
            labelRenderedSizeThreshold: this.showLabels ? 0 : 9999,
            labelDensity: 0.12,
            labelSize: 10,
            labelColor: { color: '#9ba1b0' },
            edgeLabelSize: 9,
            edgeLabelColor: { color: '#6b7280' },
            zIndex: true,
            minCameraRatio: 0.01,
            maxCameraRatio: 20,
            nodeReducer: (node, data) => this._nodeReducer(node, data),
            edgeReducer: (edge, data) => this._edgeReducer(edge, data),
        };

        // Register curved edge program if available
        if (hasCurve) {
            const CurveProgram = window.EdgeCurvedArrowProgram || window.EdgeCurveProgram;
            sigmaSettings.edgeProgramClasses = {
                curved: CurveProgram,
            };
        }

        this.renderer = new Sigma(this.graph, this.container, sigmaSettings);

        this.renderer.on('clickNode', ({ node }) => {
            this.selectedNode = node;
            this.renderer.refresh();
            if (this.onNodeClick) this.onNodeClick(node);
        });

        this.renderer.on('enterNode', ({ node }) => {
            this.hoveredNode = node;
            this._computeReachable(node);
            this.renderer.refresh();
        });

        this.renderer.on('leaveNode', () => {
            this.hoveredNode = null;
            this.reachableNodes.clear();
            this.reachableEdges.clear();
            this.renderer.refresh();
        });

        this.renderer.on('clickStage', () => {
            this.selectedNode = null;
            this.hoveredNode = null;
            this.reachableNodes.clear();
            this.reachableEdges.clear();
            this.renderer.refresh();
        });

        this.renderer.getCamera().animatedReset({ duration: 300 });
    }

    _computeReachable(startNode) {
        this.reachableNodes.clear();
        this.reachableEdges.clear();

        if ((!this.hoverShowReachable && !this.hoverShowPassthrough) ||
            !this.graph || !this.graph.hasNode(startNode)) return;

        // Forward BFS: all nodes reachable from startNode (outgoing edges)
        const forwardVisited = new Set([startNode]);
        const forwardQueue = [startNode];
        while (forwardQueue.length > 0) {
            const current = forwardQueue.shift();
            this.graph.forEachOutEdge(current, (edge, attrs, source, target) => {
                this.reachableEdges.add(edge);
                if (!forwardVisited.has(target)) {
                    forwardVisited.add(target);
                    forwardQueue.push(target);
                }
            });
        }

        if (this.hoverShowReachable) {
            // Forward-only mode
            this.reachableNodes = forwardVisited;
            return;
        }

        // Pass-through mode: also do backward BFS (incoming edges)
        const backwardVisited = new Set([startNode]);
        const backwardQueue = [startNode];
        while (backwardQueue.length > 0) {
            const current = backwardQueue.shift();
            this.graph.forEachInEdge(current, (edge, attrs, source, target) => {
                this.reachableEdges.add(edge);
                if (!backwardVisited.has(source)) {
                    backwardVisited.add(source);
                    backwardQueue.push(source);
                }
            });
        }

        // Union of forward and backward
        this.reachableNodes = new Set([...forwardVisited, ...backwardVisited]);
    }

    _hashPos(str, seed) {
        let h = seed * 13;
        for (let i = 0; i < str.length; i++) {
            h = ((h << 5) - h + str.charCodeAt(i)) | 0;
        }
        return (h % 1000) / 100;
    }

    _nodeReducer(node, data) {
        const res = { ...data };

        // Path highlighting takes priority
        if (this.highlightedPath.size > 0) {
            if (this.highlightedPath.has(node)) {
                res.color = '#f59e0b';
                res.size = data.size * 1.8;
                res.zIndex = 2;
            } else {
                res.color = '#2d3240';
                res.size = data.size * 0.6;
            }
            return res;
        }

        if (this.hoveredNode || this.selectedNode) {
            const active = this.hoveredNode || this.selectedNode;

            if (node === active) {
                // The hovered/selected node itself
                res.highlighted = true;
                res.size = data.size * 1.5;
                res.zIndex = 2;
            } else if ((this.hoverShowReachable || this.hoverShowPassthrough) && this.reachableNodes.size > 0) {
                // Reachable / pass-through mode: highlight all BFS-reachable nodes
                if (this.reachableNodes.has(node)) {
                    res.size = data.size * 1.1;
                    res.zIndex = 1;
                } else {
                    res.color = '#2d3240';
                    res.size = data.size * 0.4;
                }
            } else if (!this.hoverShowReachable && !this.hoverShowPassthrough) {
                // Direct neighbors mode
                if (this.graph.hasEdge(active, node) || this.graph.hasEdge(node, active)) {
                    res.size = data.size * 1.2;
                    res.zIndex = 1;
                } else {
                    res.color = '#2d3240';
                    res.size = data.size * 0.6;
                }
            }
        }
        return res;
    }

    _edgeReducer(edge, data) {
        const res = { ...data };

        if (!this.showEdges && !this.highlightedPathEdges.size && !this.reachableEdges.size) {
            res.hidden = true;
            return res;
        }

        // Path highlighting takes priority
        if (this.highlightedPathEdges.size > 0) {
            if (this.highlightedPathEdges.has(edge)) {
                res.color = '#f59e0b';
                res.size = 2;
                res.zIndex = 2;
                res.forceLabel = true;
            } else {
                res.hidden = true;
            }
            return res;
        }

        if (this.hoveredNode || this.selectedNode) {
            const active = this.hoveredNode || this.selectedNode;

            if ((this.hoverShowReachable || this.hoverShowPassthrough) && this.reachableEdges.size > 0) {
                // Reachable / pass-through mode: show all edges in the reachable subgraph
                if (this.reachableEdges.has(edge)) {
                    res.size = 1;
                    res.zIndex = 1;
                } else {
                    res.hidden = true;
                }
            } else if (!this.hoverShowReachable && !this.hoverShowPassthrough) {
                // Direct neighbors mode
                const source = this.graph.source(edge);
                const target = this.graph.target(edge);
                if (source === active || target === active) {
                    res.size = 1.5;
                    res.zIndex = 1;
                    res.forceLabel = true;
                } else {
                    res.hidden = true;
                }
            }
        }
        return res;
    }

    setColorMode(mode) {
        this.colorMode = mode;
        if (!this.graph || !this.currentData) return;
        const maxDegree = Math.max(1, ...this.currentData.nodes.map(n => n.degree));

        this.graph.forEachNode((node, attrs) => {
            const color = mode === 'component'
                ? COMPONENT_COLORS[attrs.sub_component % COMPONENT_COLORS.length]
                : getDegreeColor(attrs.degree, maxDegree);
            this.graph.setNodeAttribute(node, 'color', color);
            this.graph.setNodeAttribute(node, 'originalColor', color);
        });
        if (this.renderer) this.renderer.refresh();
    }

    setColorByGroup(fieldPath) {
        this.colorMode = 'group';
        if (!this.graph || !this.currentData) return;

        // Build node->value lookup and value->color index
        const nodeValueMap = new Map();
        const valueColorMap = new Map();
        let idx = 0;
        for (const n of this.currentData.nodes) {
            const val = String(n[fieldPath] ?? '');
            nodeValueMap.set(n.id, val);
            if (!valueColorMap.has(val)) {
                valueColorMap.set(val, idx++);
            }
        }

        this.graph.forEachNode((node) => {
            const val = nodeValueMap.get(node) ?? '';
            const colorIdx = valueColorMap.get(val) ?? 0;
            const color = COMPONENT_COLORS[colorIdx % COMPONENT_COLORS.length];
            this.graph.setNodeAttribute(node, 'color', color);
            this.graph.setNodeAttribute(node, 'originalColor', color);
        });
        if (this.renderer) this.renderer.refresh();
    }

    setShowEdges(show) {
        this.showEdges = show;
        if (this.renderer) this.renderer.refresh();
    }

    setShowLabels(show) {
        this.showLabels = show;
        if (this.renderer) {
            this.renderer.setSetting('labelRenderedSizeThreshold', show ? 0 : 9999);
        }
    }

    setShowEdgeLabels(show) {
        this.showEdgeLabels = show;
        if (this.renderer) {
            this.renderer.setSetting('renderEdgeLabels', show);
        }
    }

    setHoverShowReachable(show) {
        this.hoverShowReachable = show;
        if (show) this.hoverShowPassthrough = false;
    }

    setHoverShowPassthrough(show) {
        this.hoverShowPassthrough = show;
        if (show) this.hoverShowReachable = false;
    }

    highlightPath(nodeIds, edgeData) {
        this.highlightedPath = new Set(nodeIds);
        this.highlightedPathEdges.clear();
        if (edgeData && this.graph) {
            for (let i = 0; i < nodeIds.length - 1; i++) {
                const edges = this.graph.edges(nodeIds[i], nodeIds[i + 1]);
                if (edges.length > 0) this.highlightedPathEdges.add(edges[0]);
            }
        }
        if (this.renderer) this.renderer.refresh();
    }

    clearPathHighlight() {
        this.highlightedPath.clear();
        this.highlightedPathEdges.clear();
        if (this.renderer) this.renderer.refresh();
    }

    focusNode(nodeId) {
        if (!this.renderer || !this.graph || !this.graph.hasNode(nodeId)) return;
        // Just highlight the node, don't move camera
        this.selectedNode = nodeId;
        this.renderer.refresh();
    }

    destroy() {
        if (this.renderer) { this.renderer.kill(); this.renderer = null; }
        this.graph = null;
    }
}
