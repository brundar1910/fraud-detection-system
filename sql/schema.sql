-- =============================================================================
-- FraudShield AI: Real-Time Financial Fraud Detection & Investigation Platform
-- Comprehensive PostgreSQL Schema (PostgreSQL 13 - 17 Compatible)
-- Includes: Normalized Relational Tables, Constraints, Performance Indexes,
-- Machine Learning Predictions, Forensic Reports, and Analytical Views.
-- =============================================================================

-- Enable pgcrypto for UUID generation functions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- DROP EXISTING VIEWS & TABLES (Reverse Dependency Order)
-- =============================================================================
DROP VIEW IF EXISTS vw_merchant_risk_summary CASCADE;
DROP VIEW IF EXISTS vw_customer_risk_scores CASCADE;
DROP VIEW IF EXISTS vw_flagged_transactions CASCADE;

DROP TABLE IF EXISTS investigation_reports CASCADE;
DROP TABLE IF EXISTS fraud_predictions CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS fraud_alerts CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS payment_methods CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Drop trigger function if existing
DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;

-- =============================================================================
-- UTILITY FUNCTION: Automatically update updated_at timestamp
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 1. CUSTOMERS TABLE
-- Stores master customer identities, verification status, and baseline risk score
-- =============================================================================
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone_number VARCHAR(30),
    date_of_birth DATE NOT NULL,
    ssn_hash VARCHAR(128),
    address_line1 VARCHAR(255) NOT NULL,
    address_line2 VARCHAR(255),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    country VARCHAR(3) NOT NULL DEFAULT 'USA',
    risk_score NUMERIC(5, 2) NOT NULL DEFAULT 0.00
        CHECK (risk_score >= 0.00 AND risk_score <= 100.00),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
        CHECK (status IN ('ACTIVE', 'SUSPENDED', 'BLOCKED', 'CLOSED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER trg_customers_updated_at
BEFORE UPDATE ON customers
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 2. ACCOUNTS TABLE
-- Financial accounts held by customers (Checking, Savings, Credit, Investment)
-- =============================================================================
CREATE TABLE accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    account_number VARCHAR(34) NOT NULL UNIQUE,
    account_type VARCHAR(20) NOT NULL
        CHECK (account_type IN ('CHECKING', 'SAVINGS', 'CREDIT', 'INVESTMENT')),
    balance NUMERIC(15, 2) NOT NULL DEFAULT 0.00,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
        CHECK (status IN ('ACTIVE', 'FROZEN', 'DORMANT', 'CLOSED')),
    daily_limit NUMERIC(15, 2) NOT NULL DEFAULT 10000.00
        CHECK (daily_limit >= 0.00),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER trg_accounts_updated_at
BEFORE UPDATE ON accounts
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 3. MERCHANTS TABLE
-- Vendor and merchant registry with Merchant Category Code (MCC) & risk tiers
-- =============================================================================
CREATE TABLE merchants (
    merchant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_name VARCHAR(255) NOT NULL,
    category_code VARCHAR(10) NOT NULL, -- ISO 18245 MCC
    category_name VARCHAR(100) NOT NULL,
    risk_level VARCHAR(20) NOT NULL DEFAULT 'LOW'
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    country VARCHAR(3) NOT NULL DEFAULT 'USA',
    city VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
        CHECK (status IN ('ACTIVE', 'SUSPENDED', 'BLACKLISTED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER trg_merchants_updated_at
BEFORE UPDATE ON merchants
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 4. PAYMENT METHODS TABLE
-- Tokenized cards, ACH, wire, and digital wallet payment instruments
-- =============================================================================
CREATE TABLE payment_methods (
    payment_method_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    payment_type VARCHAR(30) NOT NULL
        CHECK (payment_type IN ('DEBIT_CARD', 'CREDIT_CARD', 'WIRE_TRANSFER', 'ACH', 'CRYPTO', 'DIGITAL_WALLET')),
    card_last_four VARCHAR(4),
    card_brand VARCHAR(50),
    token_reference VARCHAR(255) NOT NULL UNIQUE,
    expiry_month INT CHECK (expiry_month BETWEEN 1 AND 12),
    expiry_year INT CHECK (expiry_year >= 2020),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    billing_postal_code VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER trg_payment_methods_updated_at
BEFORE UPDATE ON payment_methods
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 5. TRANSACTIONS TABLE
-- Financial ledger capturing real-time transaction events, telemetry, & PaySim compatibility
-- =============================================================================
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
    merchant_id UUID REFERENCES merchants(merchant_id) ON DELETE SET NULL,
    payment_method_id UUID REFERENCES payment_methods(payment_method_id) ON DELETE SET NULL,
    amount NUMERIC(15, 2) NOT NULL CHECK (amount > 0.00),
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    transaction_type VARCHAR(30) NOT NULL
        CHECK (transaction_type IN ('PURCHASE', 'TRANSFER', 'WITHDRAWAL', 'DEPOSIT', 'REFUND', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT')),
    channel VARCHAR(20) NOT NULL
        CHECK (channel IN ('ONLINE', 'IN_STORE', 'ATM', 'MOBILE_APP', 'API', 'SYSTEM')),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'AUTHORIZED', 'COMPLETED', 'DECLINED', 'FLAGGED', 'REVERSED')),
    ip_address VARCHAR(45),
    device_id VARCHAR(100),
    location_lat NUMERIC(9, 6),
    location_lon NUMERIC(9, 6),
    -- PaySim-style financial simulation and auditing attributes
    step INT DEFAULT 1,
    oldbalance_orig NUMERIC(15, 2),
    newbalance_orig NUMERIC(15, 2),
    oldbalance_dest NUMERIC(15, 2),
    newbalance_dest NUMERIC(15, 2),
    name_orig VARCHAR(64),
    name_dest VARCHAR(64),
    is_fraud BOOLEAN NOT NULL DEFAULT FALSE,
    is_flagged_fraud BOOLEAN NOT NULL DEFAULT FALSE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 6. FRAUD ALERTS TABLE
-- Rule/Heuristic engine alerts flagged during transaction scoring
-- =============================================================================
CREATE TABLE fraud_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
    rule_name VARCHAR(100) NOT NULL,
    rule_description TEXT,
    risk_score NUMERIC(5, 2) NOT NULL
        CHECK (risk_score >= 0.00 AND risk_score <= 100.00),
    severity VARCHAR(20) NOT NULL
        CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN'
        CHECK (status IN ('OPEN', 'INVESTIGATING', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'RESOLVED')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT
);

-- =============================================================================
-- 7. MODEL VERSIONS TABLE
-- Tracks machine learning model metadata, training lineage, and evaluation metrics
-- =============================================================================
CREATE TABLE model_versions (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL UNIQUE,
    algorithm VARCHAR(100) NOT NULL,
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    feature_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    artifact_path VARCHAR(255),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    training_samples_count INT,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 8. FRAUD PREDICTIONS TABLE
-- Machine learning inference logs with model version, risk tier, and XAI/SHAP values
-- =============================================================================
CREATE TABLE fraud_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL REFERENCES model_versions(version) ON UPDATE CASCADE,
    fraud_probability NUMERIC(5, 4) NOT NULL
        CHECK (fraud_probability >= 0.0000 AND fraud_probability <= 1.0000),
    prediction_label VARCHAR(20) NOT NULL
        CHECK (prediction_label IN ('LEGITIMATE', 'SUSPICIOUS', 'FRAUD')),
    risk_tier VARCHAR(20) NOT NULL
        CHECK (risk_tier IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    top_features JSONB NOT NULL DEFAULT '{}'::jsonb, -- e.g. {"amount": 0.42, "velocity_10m": 0.31}
    inference_latency_ms NUMERIC(8, 2),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 9. INVESTIGATION REPORTS TABLE
-- Forensic case files, graph cluster linkages, and autonomous AI narratives
-- =============================================================================
CREATE TABLE investigation_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id UUID REFERENCES fraud_alerts(alert_id) ON DELETE SET NULL,
    transaction_id UUID REFERENCES transactions(transaction_id) ON DELETE SET NULL,
    account_id UUID REFERENCES accounts(account_id) ON DELETE SET NULL,
    investigator_id VARCHAR(100) NOT NULL DEFAULT 'system_ai',
    report_title VARCHAR(255) NOT NULL,
    executive_summary TEXT NOT NULL,
    evidence_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    graph_cluster_id VARCHAR(100),
    genai_narrative TEXT, -- Narrative autonomously written by Google GenAI
    recommended_action VARCHAR(50) NOT NULL
        CHECK (recommended_action IN ('FREEZE_ACCOUNT', 'BLOCK_MERCHANT', 'CANCEL_CARD', 'REQUEST_STEP_UP_AUTH', 'DISMISS_FALSE_POSITIVE', 'FILE_SAR')),
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT'
        CHECK (status IN ('DRAFT', 'PENDING_REVIEW', 'APPROVED', 'CLOSED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER trg_investigation_reports_updated_at
BEFORE UPDATE ON investigation_reports
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- PERFORMANCE & TIME-SERIES INDEXES
-- Optimized for high-throughput streaming lookups, aggregations, and velocity checks
-- =============================================================================

-- Indexes on account_id
CREATE INDEX idx_accounts_customer_id ON accounts(customer_id);
CREATE INDEX idx_payment_methods_account_id ON payment_methods(account_id);
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_fraud_alerts_account_id ON fraud_alerts(account_id);
CREATE INDEX idx_investigation_reports_account_id ON investigation_reports(account_id);

-- Indexes on timestamps
CREATE INDEX idx_customers_created_at ON customers(created_at);
CREATE INDEX idx_accounts_created_at ON accounts(created_at);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_fraud_alerts_created_at ON fraud_alerts(created_at);
CREATE INDEX idx_fraud_predictions_predicted_at ON fraud_predictions(predicted_at);
CREATE INDEX idx_investigation_reports_created_at ON investigation_reports(created_at);

-- Composite Index for Real-Time Velocity / Time-Window Fraud Analysis
-- (e.g. Find all transactions on account X within the last N minutes)
CREATE INDEX idx_transactions_account_timestamp ON transactions(account_id, timestamp DESC);
CREATE INDEX idx_fraud_alerts_account_created ON fraud_alerts(account_id, created_at DESC);

-- Fraud Alert & Prediction Filtering Indexes
CREATE INDEX idx_fraud_alerts_transaction_id ON fraud_alerts(transaction_id);
CREATE INDEX idx_fraud_alerts_status_severity ON fraud_alerts(status, severity);
CREATE INDEX idx_fraud_predictions_transaction_id ON fraud_predictions(transaction_id);
CREATE INDEX idx_fraud_predictions_tier ON fraud_predictions(risk_tier);
CREATE INDEX idx_model_versions_active ON model_versions(is_active);

-- Auxiliary Indexes for Merchant & Transaction Lookups
CREATE INDEX idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_merchants_category_code ON merchants(category_code);
CREATE INDEX idx_transactions_is_fraud ON transactions(is_fraud);


-- =============================================================================
-- ANALYTICAL SQL VIEWS
-- Using CTEs, Window Functions (LAG, rolling AVG/COUNT), Subqueries, and Multi-Table Joins
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. VIEW: vw_flagged_transactions
-- Real-time transaction surveillance view computing temporal gaps (LAG), rolling
-- velocity/amount averages, spend spike ratios, and linked alerts & ML predictions.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_flagged_transactions AS
WITH tx_temporal_metrics AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.merchant_id,
        t.amount,
        t.currency,
        t.transaction_type,
        t.channel,
        t.status,
        t.timestamp,
        t.is_fraud AS ground_truth_fraud,
        t.ip_address,
        t.device_id,
        -- Previous transaction details via LAG
        LAG(t.timestamp) OVER (
            PARTITION BY t.account_id 
            ORDER BY t.timestamp
        ) AS prev_txn_timestamp,
        LAG(t.amount) OVER (
            PARTITION BY t.account_id 
            ORDER BY t.timestamp
        ) AS prev_txn_amount,
        LAG(t.merchant_id) OVER (
            PARTITION BY t.account_id 
            ORDER BY t.timestamp
        ) AS prev_merchant_id,
        -- Rolling historical average ticket size for this account (10 preceding txns)
        AVG(t.amount) OVER (
            PARTITION BY t.account_id 
            ORDER BY t.timestamp 
            ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_amount_prior,
        -- Rolling count of transactions in the preceding 5 events
        COUNT(*) OVER (
            PARTITION BY t.account_id 
            ORDER BY t.timestamp 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS recent_burst_count
    FROM transactions t
),
enriched_tx AS (
    SELECT
        m.*,
        EXTRACT(EPOCH FROM (m.timestamp - m.prev_txn_timestamp)) AS seconds_since_prev_txn,
        CASE 
            WHEN m.rolling_avg_amount_prior IS NOT NULL AND m.rolling_avg_amount_prior > 0 
            THEN ROUND((m.amount / m.rolling_avg_amount_prior), 2)
            ELSE 1.00 
        END AS spending_spike_ratio,
        CASE 
            WHEN m.prev_merchant_id IS NOT NULL 
                 AND m.merchant_id = m.prev_merchant_id 
                 AND EXTRACT(EPOCH FROM (m.timestamp - m.prev_txn_timestamp)) <= 120
                 AND m.amount = m.prev_txn_amount
            THEN TRUE
            ELSE FALSE
        END AS is_duplicate_burst
    FROM tx_temporal_metrics m
)
SELECT
    e.transaction_id,
    e.account_id,
    c.customer_id,
    c.first_name || ' ' || c.last_name AS customer_name,
    c.email AS customer_email,
    c.risk_score AS customer_base_risk,
    a.account_number,
    a.account_type,
    e.merchant_id,
    mer.merchant_name,
    mer.category_code AS merchant_mcc,
    mer.risk_level AS merchant_risk_level,
    e.amount,
    e.currency,
    e.transaction_type,
    e.channel,
    e.status,
    e.timestamp,
    e.seconds_since_prev_txn,
    e.spending_spike_ratio,
    e.recent_burst_count,
    e.is_duplicate_burst,
    e.ground_truth_fraud,
    -- Linked fraud alerts aggregation
    COUNT(fa.alert_id) AS total_rules_triggered,
    MAX(fa.risk_score) AS max_rule_risk_score,
    MAX(fa.severity) AS max_rule_severity,
    -- Linked machine learning predictions
    fp.fraud_probability AS ml_fraud_probability,
    fp.risk_tier AS ml_risk_tier,
    fp.top_features AS ml_top_features,
    -- High-level composite flag
    CASE
        WHEN e.ground_truth_fraud = TRUE THEN 'CONFIRMED_FRAUD'
        WHEN fp.fraud_probability >= 0.85 OR MAX(fa.risk_score) >= 85.0 THEN 'CRITICAL_SUSPECT'
        WHEN e.is_duplicate_burst = TRUE OR e.spending_spike_ratio >= 5.0 THEN 'PATTERN_ANOMALY'
        WHEN COUNT(fa.alert_id) > 0 THEN 'FLAGGED'
        ELSE 'NORMAL'
    END AS surveillance_verdict
FROM enriched_tx e
JOIN accounts a ON e.account_id = a.account_id
JOIN customers c ON a.customer_id = c.customer_id
LEFT JOIN merchants mer ON e.merchant_id = mer.merchant_id
LEFT JOIN fraud_alerts fa ON e.transaction_id = fa.transaction_id
LEFT JOIN fraud_predictions fp ON e.transaction_id = fp.transaction_id
GROUP BY
    e.transaction_id, e.account_id, c.customer_id, c.first_name, c.last_name,
    c.email, c.risk_score, a.account_number, a.account_type, e.merchant_id,
    mer.merchant_name, mer.category_code, mer.risk_level, e.amount, e.currency,
    e.transaction_type, e.channel, e.status, e.timestamp, e.seconds_since_prev_txn,
    e.spending_spike_ratio, e.recent_burst_count, e.is_duplicate_burst,
    e.ground_truth_fraud, fp.fraud_probability, fp.risk_tier, fp.top_features;


-- -----------------------------------------------------------------------------
-- 2. VIEW: vw_customer_risk_scores
-- Comprehensive customer risk profile aggregating account portfolios, transaction
-- volume, alert frequency, and comparative risk ranking across the portfolio.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_customer_risk_scores AS
WITH customer_tx_stats AS (
    SELECT
        c.customer_id,
        COUNT(DISTINCT a.account_id) AS active_accounts_count,
        COALESCE(SUM(a.balance), 0.00) AS total_balance,
        COUNT(t.transaction_id) AS total_transactions,
        COALESCE(SUM(t.amount), 0.00) AS total_spent,
        COALESCE(AVG(t.amount), 0.00) AS avg_transaction_amount,
        COALESCE(MAX(t.amount), 0.00) AS max_single_transaction,
        COUNT(CASE WHEN t.status IN ('FLAGGED', 'DECLINED') THEN 1 END) AS anomalous_transactions_count,
        COUNT(CASE WHEN t.is_fraud = TRUE THEN 1 END) AS verified_fraud_count
    FROM customers c
    LEFT JOIN accounts a ON c.customer_id = a.customer_id
    LEFT JOIN transactions t ON a.account_id = t.account_id
    GROUP BY c.customer_id
),
customer_alert_stats AS (
    SELECT
        c.customer_id,
        COUNT(fa.alert_id) AS total_alerts_raised,
        COUNT(CASE WHEN fa.severity = 'CRITICAL' THEN 1 END) AS critical_alerts_count,
        COUNT(CASE WHEN fa.severity = 'HIGH' THEN 1 END) AS high_alerts_count,
        COALESCE(MAX(fa.risk_score), 0.00) AS peak_alert_risk_score
    FROM customers c
    LEFT JOIN accounts a ON c.customer_id = a.customer_id
    LEFT JOIN fraud_alerts fa ON a.account_id = fa.account_id
    GROUP BY c.customer_id
),
scored_customers AS (
    SELECT
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        c.status AS customer_status,
        c.risk_score AS base_kyc_risk,
        ts.active_accounts_count,
        ts.total_balance,
        ts.total_transactions,
        ts.total_spent,
        ROUND(ts.avg_transaction_amount, 2) AS avg_transaction_amount,
        ts.max_single_transaction,
        ts.anomalous_transactions_count,
        ts.verified_fraud_count,
        als.total_alerts_raised,
        als.critical_alerts_count,
        als.high_alerts_count,
        als.peak_alert_risk_score,
        -- Composite Risk Score Formula
        LEAST(100.00, ROUND(
            (c.risk_score * 0.20) +
            (LEAST(ts.anomalous_transactions_count * 15.0, 30.0)) +
            (LEAST(als.critical_alerts_count * 25.0 + als.high_alerts_count * 10.0, 40.0)) +
            (CASE WHEN ts.verified_fraud_count > 0 THEN 25.0 ELSE 0.0 END),
            2
        )) AS calculated_composite_risk
    FROM customers c
    JOIN customer_tx_stats ts ON c.customer_id = ts.customer_id
    JOIN customer_alert_stats als ON c.customer_id = als.customer_id
)
SELECT
    sc.*,
    -- Rank customers based on composite risk
    DENSE_RANK() OVER (ORDER BY sc.calculated_composite_risk DESC, sc.total_alerts_raised DESC) AS risk_rank,
    ROUND(PERCENT_RANK() OVER (ORDER BY sc.calculated_composite_risk)::NUMERIC, 4) AS risk_percentile,
    CASE
        WHEN sc.calculated_composite_risk >= 80.00 THEN 'CRITICAL'
        WHEN sc.calculated_composite_risk >= 50.00 THEN 'HIGH'
        WHEN sc.calculated_composite_risk >= 25.00 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_tier
FROM scored_customers sc;


-- -----------------------------------------------------------------------------
-- 3. VIEW: vw_merchant_risk_summary
-- Aggregated vendor risk telemetry ranking merchants within MCC categories
-- using category rolling averages, fraud incidence ratios, and percentile ranks.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_merchant_risk_summary AS
WITH merchant_metrics AS (
    SELECT
        m.merchant_id,
        m.merchant_name,
        m.category_code,
        m.category_name,
        m.risk_level AS registered_risk_level,
        m.status AS merchant_status,
        m.country,
        m.city,
        COUNT(t.transaction_id) AS total_transactions_processed,
        COALESCE(SUM(t.amount), 0.00) AS total_settled_volume,
        COALESCE(AVG(t.amount), 0.00) AS avg_ticket_size,
        COUNT(CASE WHEN t.status IN ('FLAGGED', 'DECLINED', 'REVERSED') THEN 1 END) AS disputed_or_declined_txns,
        COUNT(CASE WHEN t.is_fraud = TRUE THEN 1 END) AS confirmed_fraud_txns,
        COUNT(fa.alert_id) AS total_fraud_alerts_incurred
    FROM merchants m
    LEFT JOIN transactions t ON m.merchant_id = t.merchant_id
    LEFT JOIN fraud_alerts fa ON t.transaction_id = fa.transaction_id
    GROUP BY m.merchant_id
),
calculated_rates AS (
    SELECT
        mm.*,
        CASE
            WHEN mm.total_transactions_processed > 0
            THEN ROUND((mm.disputed_or_declined_txns::NUMERIC / mm.total_transactions_processed) * 100, 2)
            ELSE 0.00
        END AS dispute_rate_pct,
        CASE
            WHEN mm.total_transactions_processed > 0
            THEN ROUND((mm.total_fraud_alerts_incurred::NUMERIC / mm.total_transactions_processed) * 100, 2)
            ELSE 0.00
        END AS alert_rate_pct
    FROM merchant_metrics mm
)
SELECT
    cr.merchant_id,
    cr.merchant_name,
    cr.category_code,
    cr.category_name,
    cr.registered_risk_level,
    cr.merchant_status,
    cr.country,
    cr.city,
    cr.total_transactions_processed,
    ROUND(cr.total_settled_volume, 2) AS total_settled_volume,
    ROUND(cr.avg_ticket_size, 2) AS avg_ticket_size,
    cr.disputed_or_declined_txns,
    cr.confirmed_fraud_txns,
    cr.total_fraud_alerts_incurred,
    cr.dispute_rate_pct,
    cr.alert_rate_pct,
    -- Window function: Category benchmark average ticket size
    ROUND(AVG(cr.avg_ticket_size) OVER (PARTITION BY cr.category_code), 2) AS category_avg_ticket_size,
    -- Window function: Risk ranking within category
    DENSE_RANK() OVER (
        PARTITION BY cr.category_code 
        ORDER BY cr.alert_rate_pct DESC, cr.disputed_or_declined_txns DESC
    ) AS category_risk_rank,
    -- Global fraud exposure percentile across all merchants
    ROUND(PERCENT_RANK() OVER (ORDER BY cr.alert_rate_pct, cr.total_fraud_alerts_incurred)::NUMERIC, 4) AS global_fraud_percentile,
    -- Real-time dynamic risk classification
    CASE
        WHEN cr.alert_rate_pct >= 10.0 OR cr.confirmed_fraud_txns >= 5 THEN 'CRITICAL'
        WHEN cr.alert_rate_pct >= 4.0 OR cr.dispute_rate_pct >= 15.0 THEN 'HIGH'
        WHEN cr.alert_rate_pct >= 1.0 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS operational_risk_tier
FROM calculated_rates cr;
