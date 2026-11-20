"""
FraudShield AI - FastAPI Request and Response Pydantic Schemas
Defines structured validation contracts for all API endpoints.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Transaction Scoring Request & Response
# -----------------------------------------------------------------------------
class TransactionScoreRequest(BaseModel):
    account_id: str = Field(..., description="UUID or ID of the originating account")
    amount: float = Field(..., gt=0, description="Monetary transaction amount in USD")
    transaction_type: str = Field("PAYMENT", description="PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN")
    channel: str = Field("ONLINE", description="ONLINE, IN_STORE, ATM, MOBILE_APP, API")
    merchant_id: Optional[str] = Field(None, description="Target merchant UUID (if applicable)")
    oldbalance_orig: Optional[float] = Field(None, description="Sender balance prior to transaction")
    newbalance_orig: Optional[float] = Field(None, description="Sender balance following transaction")
    oldbalance_dest: Optional[float] = Field(0.0, description="Recipient balance prior to transaction")
    newbalance_dest: Optional[float] = Field(0.0, description="Recipient balance following transaction")
    name_orig: Optional[str] = Field(None, description="Sender account number or identifier")
    name_dest: Optional[str] = Field(None, description="Recipient account number or merchant identifier")
    ip_address: Optional[str] = Field(None, description="Client IPv4 or IPv6 address")
    device_id: Optional[str] = Field(None, description="Client hardware device fingerprint hash")


class TransactionScoreResponse(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    fraud_probability: float
    prediction_label: str  # FRAUD, SUSPICIOUS, LEGITIMATE
    risk_tier: str         # CRITICAL, HIGH, MEDIUM, LOW
    is_flagged: bool
    status: str            # COMPLETED, FLAGGED, DECLINED
    rule_alerts_triggered: List[str]
    top_shap_features: Dict[str, float]
    inference_latency_ms: float
    surveillance_verdict: str


# -----------------------------------------------------------------------------
# System & KPI Metrics Summary
# -----------------------------------------------------------------------------
class MetricsSummaryResponse(BaseModel):
    total_transactions: int
    total_fraud_alerts: int
    high_risk_customers_count: int
    confirmed_fraud_count: int
    fraud_detection_rate_pct: float
    active_model_name: str
    active_model_version: str
    model_pr_auc: float
    model_roc_auc: float
    total_settled_volume_usd: float
    total_fraud_volume_usd: float


# -----------------------------------------------------------------------------
# Item Schemas (Transactions, Alerts, Customers)
# -----------------------------------------------------------------------------
class TransactionListItem(BaseModel):
    transaction_id: str
    account_id: str
    customer_name: Optional[str] = None
    merchant_name: Optional[str] = None
    amount: float
    transaction_type: str
    channel: str
    status: str
    is_fraud: bool
    timestamp: str
    spending_spike_ratio: Optional[float] = None
    surveillance_verdict: Optional[str] = None


class AlertListItem(BaseModel):
    alert_id: str
    transaction_id: str
    account_id: str
    rule_name: str
    rule_description: Optional[str] = None
    risk_score: float
    severity: str
    status: str
    metadata: Dict[str, Any]
    created_at: str


class CustomerRiskProfileResponse(BaseModel):
    customer_id: str
    first_name: str
    last_name: str
    email: str
    customer_status: str
    base_kyc_risk: float
    calculated_composite_risk: float
    risk_tier: str
    risk_rank: int
    risk_percentile: float
    total_transactions: int
    total_spent: float
    active_accounts_count: int
    total_balance: float
    anomalous_transactions_count: int
    verified_fraud_count: int
    total_alerts_raised: int
    critical_alerts_count: int
    recent_transactions: List[Dict[str, Any]]
    associated_alerts: List[Dict[str, Any]]
    shared_infrastructure: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Network Graph Response
# -----------------------------------------------------------------------------
class NetworkGraphNode(BaseModel):
    id: str
    label: str
    type: str
    color: str


class NetworkGraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    amount: float
    is_fraud: bool


class NetworkGraphResponse(BaseModel):
    total_nodes: int
    total_edges: int
    shared_infrastructure_count: int
    merchant_fraud_rings_count: int
    nodes: List[NetworkGraphNode]
    edges: List[NetworkGraphEdge]


# -----------------------------------------------------------------------------
# Forensic Investigation Agent Response
# -----------------------------------------------------------------------------
class InvestigationReportResponse(BaseModel):
    report_id: str
    transaction_id: str
    account_id: Optional[str] = None
    investigator_id: str
    report_title: str
    executive_summary: str
    recommended_action: str
    status: str
    genai_narrative: str
    evidence_summary: Dict[str, Any]
    created_at: str
