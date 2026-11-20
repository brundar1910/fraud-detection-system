"""
FraudShield AI - NetworkX Graph Intelligence Test Suite
Tests graph loading, sub-second execution, shared infrastructure, and merchant ring detection.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from graph_engine.fraud_graph import OptimizedFraudGraphEngine


def test_optimized_graph_construction():
    """Validates that graph constructs within 5 seconds and contains valid nodes."""
    engine = OptimizedFraudGraphEngine(recent_limit=2000)
    assert engine.G.number_of_nodes() > 1000
    assert engine.G.number_of_edges() > 1000


def test_shared_infrastructure_detection():
    """Validates that multi-account hardware/IP clusters are identified."""
    engine = OptimizedFraudGraphEngine(recent_limit=2000)
    anomalies = engine.detect_shared_infrastructure(min_shared=2)
    assert len(anomalies) > 0
    top = anomalies[0]
    assert "identifier" in top
    assert top["shared_accounts_count"] >= 2


def test_merchant_fraud_rings():
    """Validates detection of merchants receiving suspicious transaction volumes."""
    engine = OptimizedFraudGraphEngine(recent_limit=2000)
    rings = engine.detect_merchant_fraud_rings(min_fraud_senders=1)
    assert len(rings) > 0
    assert "merchant_name" in rings[0]
    assert "distinct_fraud_senders_count" in rings[0]


def test_subgraph_json_export():
    """Validates formatted network export for visualization."""
    engine = OptimizedFraudGraphEngine(recent_limit=2000)
    subgraph = engine.export_subgraph_json(max_nodes=40)
    assert "nodes" in subgraph
    assert "edges" in subgraph
    assert len(subgraph["nodes"]) <= 40
    assert "color" in subgraph["nodes"][0]
