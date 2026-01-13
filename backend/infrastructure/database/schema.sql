-- Medical AI Platform Database Schema
-- HIPAA Compliant with Comprehensive Audit Logging

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ===========================================
-- PATIENTS TABLE (HIPAA Compliant)
-- ===========================================

CREATE TABLE patients (
    patient_id_hash VARCHAR(64) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Demographics (aggregated, not direct identifiers)
    age_group VARCHAR(20) CHECK (age_group IN ('18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+')),
    gender_category VARCHAR(10) CHECK (gender_category IN ('M', 'F', 'Other', 'Unknown')),
    ethnicity_category VARCHAR(50),

    -- Clinical metadata (de-identified)
    risk_factors JSONB DEFAULT '{}',
    clinical_history_summary TEXT,
    family_history_relevant BOOLEAN DEFAULT FALSE,

    -- System metadata
    data_source VARCHAR(100),
    consent_status VARCHAR(20) DEFAULT 'active' CHECK (consent_status IN ('active', 'revoked', 'expired')),
    retention_period_months INTEGER DEFAULT 84, -- 7 years HIPAA requirement

    CONSTRAINT valid_hash CHECK (LENGTH(patient_id_hash) = 64)
);

-- Indexes for patients
CREATE INDEX idx_patients_age_group ON patients(age_group);
CREATE INDEX idx_patients_gender ON patients(gender_category);
CREATE INDEX idx_patients_created ON patients(created_at);
CREATE INDEX idx_patients_consent ON patients(consent_status);

-- ===========================================
-- MEDICAL STUDIES TABLE
-- ===========================================

CREATE TABLE studies (
    study_uid VARCHAR(64) PRIMARY KEY,
    patient_id_hash VARCHAR(64) REFERENCES patients(patient_id_hash) ON DELETE CASCADE,
    study_instance_uid VARCHAR(100) UNIQUE,

    -- Study metadata
    study_date DATE NOT NULL,
    study_time TIME,
    study_description TEXT,
    accession_number VARCHAR(50),

    -- Modality and anatomy
    modality VARCHAR(16) NOT NULL CHECK (modality IN ('CT', 'MR', 'CR', 'DX', 'MG', 'US', 'NM', 'PT', 'OT')),
    body_part_examined VARCHAR(50),
    laterality VARCHAR(10) CHECK (laterality IN ('R', 'L', 'B', 'U')),

    -- Technical parameters
    kvp DECIMAL(6,2),
    exposure_time INTEGER,
    exposure DECIMAL(8,2),
    image_laterality VARCHAR(10),

    -- Clinical information (de-identified)
    clinical_indication TEXT,
    comparison_studies JSONB DEFAULT '[]',

    -- Processing metadata
    processed BOOLEAN DEFAULT FALSE,
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    processing_attempts INTEGER DEFAULT 0,
    last_processed TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_study_uid CHECK (LENGTH(study_uid) = 64)
);

-- Indexes for studies
CREATE INDEX idx_studies_patient ON studies(patient_id_hash);
CREATE INDEX idx_studies_date ON studies(study_date DESC);
CREATE INDEX idx_studies_modality ON studies(modality);
CREATE INDEX idx_studies_body_part ON studies(body_part_examined);
CREATE INDEX idx_studies_processing_status ON studies(processing_status);
CREATE INDEX idx_studies_processed ON studies(processed);

-- ===========================================
-- DICOM SERIES TABLE
-- ===========================================

CREATE TABLE series (
    series_uid VARCHAR(64) PRIMARY KEY,
    study_uid VARCHAR(64) REFERENCES studies(study_uid) ON DELETE CASCADE,
    series_instance_uid VARCHAR(100) UNIQUE,

    -- Series metadata
    series_number INTEGER,
    series_date DATE,
    series_time TIME,
    series_description TEXT,
    protocol_name VARCHAR(100),

    -- Image characteristics
    rows INTEGER,
    columns INTEGER,
    bits_allocated INTEGER,
    bits_stored INTEGER,
    high_bit INTEGER,
    pixel_representation INTEGER,
    samples_per_pixel INTEGER,
    photometric_interpretation VARCHAR(20),
    pixel_spacing VARCHAR(50),
    image_orientation VARCHAR(100),
    image_position VARCHAR(100),
    slice_thickness DECIMAL(6,2),
    spacing_between_slices DECIMAL(6,2),

    -- Storage information (encrypted)
    s3_bucket VARCHAR(100),
    s3_key VARCHAR(512),
    file_size_bytes BIGINT,
    content_hash VARCHAR(64), -- SHA256 of original file
    encryption_method VARCHAR(50) DEFAULT 'AES256',
    compression_method VARCHAR(20),

    -- Processing status
    anonymized BOOLEAN DEFAULT FALSE,
    phi_removed BOOLEAN DEFAULT FALSE,
    quality_score DECIMAL(3,2), -- 0.00 to 1.00

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_series_uid CHECK (LENGTH(series_uid) = 64),
    CONSTRAINT valid_content_hash CHECK (content_hash IS NULL OR LENGTH(content_hash) = 64)
);

-- Indexes for series
CREATE INDEX idx_series_study ON series(study_uid);
CREATE INDEX idx_series_date ON series(series_date DESC);
CREATE INDEX idx_series_content_hash ON series(content_hash);
CREATE INDEX idx_series_s3_key ON series(s3_key);

-- ===========================================
-- AI PREDICTIONS TABLE
-- ===========================================

CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_uid VARCHAR(64) REFERENCES series(series_uid) ON DELETE CASCADE,

    -- Model information
    model_version VARCHAR(32) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    cancer_type VARCHAR(32) NOT NULL,

    -- Prediction results
    prediction VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    probability_benign DECIMAL(5,4) CHECK (probability_benign >= 0 AND probability_benign <= 1),
    probability_malignant DECIMAL(5,4) CHECK (probability_malignant >= 0 AND probability_malignant <= 1),

    -- Risk assessment
    risk_level VARCHAR(16) NOT NULL CHECK (risk_level IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH')),
    risk_score DECIMAL(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    clinical_significance TEXT NOT NULL,

    -- Explainability
    xai_method VARCHAR(50) DEFAULT 'gradcam',
    attention_regions JSONB DEFAULT '{}',
    saliency_map_metadata JSONB DEFAULT '{}',

    -- Clinical workflow
    requires_followup BOOLEAN DEFAULT FALSE,
    recommended_action VARCHAR(100),
    urgency_level VARCHAR(20) DEFAULT 'routine' CHECK (urgency_level IN ('routine', 'urgent', 'asap', 'stat')),

    -- Physician review
    physician_reviewed BOOLEAN DEFAULT FALSE,
    physician_id VARCHAR(64),
    physician_notes TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    agreement_level VARCHAR(20) CHECK (agreement_level IN ('agree', 'disagree', 'uncertain')),

    -- Quality metrics
    processing_time_seconds DECIMAL(6,3),
    model_inference_time DECIMAL(6,3),
    preprocessing_time DECIMAL(6,3),

    -- Audit trail
    created_by VARCHAR(64) NOT NULL, -- User who requested prediction
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_predictions_series (series_uid),
    INDEX idx_predictions_cancer_type (cancer_type),
    INDEX idx_predictions_confidence (confidence),
    INDEX idx_predictions_risk_level (risk_level),
    INDEX idx_predictions_created (created_at),
    INDEX idx_predictions_physician_review (physician_reviewed),
    INDEX idx_predictions_user (created_by)
);

-- ===========================================
-- MODEL REGISTRY TABLE
-- ===========================================

CREATE TABLE model_registry (
    model_id VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    cancer_type VARCHAR(32) NOT NULL,
    version VARCHAR(32) NOT NULL,

    -- Model metadata
    framework VARCHAR(50) DEFAULT 'pytorch',
    input_shape JSONB DEFAULT '[224, 224, 3]',
    output_classes JSONB NOT NULL,

    -- Performance metrics
    accuracy DECIMAL(5,4),
    sensitivity DECIMAL(5,4),
    specificity DECIMAL(5,4),
    auc DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    precision_recall_auc DECIMAL(5,4),

    -- Validation results
    validation_dataset VARCHAR(100),
    validation_samples INTEGER,
    cross_validation_folds INTEGER,
    bootstrap_iterations INTEGER,
    confidence_interval_95 JSONB, -- [lower, upper] bounds

    -- Bias and fairness
    demographic_parity JSONB DEFAULT '{}',
    equal_opportunity JSONB DEFAULT '{}',
    fairness_metrics JSONB DEFAULT '{}',

    -- Storage
    s3_bucket VARCHAR(100),
    s3_model_key VARCHAR(512),
    s3_config_key VARCHAR(512),
    model_size_bytes BIGINT,

    -- Lifecycle
    is_active BOOLEAN DEFAULT FALSE,
    training_completed_at TIMESTAMP WITH TIME ZONE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(model_name, version),
    INDEX idx_model_registry_active (is_active),
    INDEX idx_model_registry_cancer_type (cancer_type),
    INDEX idx_model_registry_performance (auc DESC, f1_score DESC)
);

-- ===========================================
-- AUDIT LOG TABLE (HIPAA Required)
-- ===========================================

CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- User information
    user_id VARCHAR(64) NOT NULL,
    user_role VARCHAR(50),
    user_department VARCHAR(100),
    session_id VARCHAR(64),

    -- Action details
    action VARCHAR(64) NOT NULL, -- 'login', 'predict', 'review', 'export', etc.
    resource_type VARCHAR(32), -- 'prediction', 'study', 'model', 'report'
    resource_id VARCHAR(64),
    action_details JSONB DEFAULT '{}',

    -- Data access
    patient_id_hash VARCHAR(64), -- For tracking access patterns
    study_uid VARCHAR(64),
    series_uid VARCHAR(64),

    -- System information
    ip_address INET,
    user_agent TEXT,
    api_endpoint VARCHAR(200),
    http_method VARCHAR(10),
    response_status_code INTEGER,

    -- Security
    authentication_method VARCHAR(50) DEFAULT 'jwt',
    mfa_used BOOLEAN DEFAULT FALSE,
    suspicious_activity BOOLEAN DEFAULT FALSE,

    -- Compliance
    hipaa_purpose VARCHAR(100), -- e.g., 'treatment', 'research', 'audit'
    data_retention_category VARCHAR(50),

    INDEX idx_audit_user (user_id),
    INDEX idx_audit_timestamp (timestamp),
    INDEX idx_audit_action (action),
    INDEX idx_audit_resource (resource_type, resource_id),
    INDEX idx_audit_patient (patient_id_hash),
    INDEX idx_audit_suspicious (suspicious_activity)
);

-- ===========================================
-- CLINICAL PERFORMANCE METRICS TABLE
-- ===========================================

CREATE TABLE clinical_metrics (
    metric_date DATE NOT NULL,
    cancer_type VARCHAR(32) NOT NULL,
    model_version VARCHAR(32) NOT NULL,

    -- Volume metrics
    total_predictions INTEGER DEFAULT 0,
    reviewed_predictions INTEGER DEFAULT 0,

    -- Performance metrics
    sensitivity DECIMAL(5,4),
    specificity DECIMAL(5,4),
    positive_predictive_value DECIMAL(5,4),
    negative_predictive_value DECIMAL(5,4),
    diagnostic_odds_ratio DECIMAL(8,4),

    -- Clinical outcomes
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,

    -- Risk stratification
    high_risk_predictions INTEGER DEFAULT 0,
    medium_risk_predictions INTEGER DEFAULT 0,
    low_risk_predictions INTEGER DEFAULT 0,

    -- Physician agreement
    physician_agreement_rate DECIMAL(5,4),
    clinical_impact_score DECIMAL(5,4),

    -- System performance
    average_processing_time DECIMAL(6,3),
    error_rate DECIMAL(5,4),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    PRIMARY KEY (metric_date, cancer_type, model_version),
    INDEX idx_clinical_metrics_date (metric_date DESC),
    INDEX idx_clinical_metrics_performance (sensitivity, specificity)
);

-- ===========================================
-- DASHBOARD MATERIALIZED VIEW
-- ===========================================

CREATE MATERIALIZED VIEW dashboard_metrics AS
SELECT
    DATE(created_at) as metric_date,
    cancer_type,
    model_version,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN physician_reviewed THEN 1 END) as reviewed_predictions,
    AVG(confidence) as avg_confidence,
    AVG(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_risk_rate,
    AVG(processing_time_seconds) as avg_processing_time,
    COUNT(CASE WHEN physician_agreement_level = 'agree' THEN 1 END)::DECIMAL /
        NULLIF(COUNT(CASE WHEN physician_reviewed THEN 1 END), 0) as physician_agreement_rate
FROM predictions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), cancer_type, model_version
ORDER BY metric_date DESC, cancer_type;

-- Refresh the materialized view every hour
CREATE OR REPLACE FUNCTION refresh_dashboard_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_metrics;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- SECURITY POLICIES (Row Level Security)
-- ===========================================

-- Enable Row Level Security
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE series ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- RLS Policies for patients
CREATE POLICY patients_access_policy ON patients
    FOR ALL USING (
        -- Users can only access patients they have permission for
        patient_id_hash IN (
            SELECT patient_id_hash FROM user_patient_permissions
            WHERE user_id = current_user_id()
        )
    );

-- RLS Policies for predictions
CREATE POLICY predictions_access_policy ON predictions
    FOR ALL USING (
        -- Users can access predictions for patients they can access
        series_uid IN (
            SELECT s.series_uid FROM series s
            JOIN studies st ON s.study_uid = st.study_uid
            WHERE st.patient_id_hash IN (
                SELECT patient_id_hash FROM user_patient_permissions
                WHERE user_id = current_user_id()
            )
        )
    );

-- ===========================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ===========================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables
CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_studies_updated_at BEFORE UPDATE ON studies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_series_updated_at BEFORE UPDATE ON series
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit trigger for predictions
CREATE OR REPLACE FUNCTION audit_prediction_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (
        user_id, action, resource_type, resource_id,
        patient_id_hash, study_uid, series_uid,
        action_details, hipaa_purpose
    )
    SELECT
        COALESCE(NEW.created_by, OLD.created_by, 'system'),
        CASE
            WHEN TG_OP = 'INSERT' THEN 'create_prediction'
            WHEN TG_OP = 'UPDATE' THEN 'update_prediction'
            WHEN TG_OP = 'DELETE' THEN 'delete_prediction'
        END,
        'prediction',
        COALESCE(NEW.prediction_id, OLD.prediction_id)::TEXT,
        (SELECT patient_id_hash FROM series s JOIN studies st ON s.study_uid = st.study_uid
         WHERE s.series_uid = COALESCE(NEW.series_uid, OLD.series_uid)),
        (SELECT study_uid FROM series WHERE series_uid = COALESCE(NEW.series_uid, OLD.series_uid)),
        COALESCE(NEW.series_uid, OLD.series_uid),
        jsonb_build_object(
            'operation', TG_OP,
            'old_values', CASE WHEN TG_OP != 'INSERT' THEN row_to_json(OLD) ELSE NULL END,
            'new_values', CASE WHEN TG_OP != 'DELETE' THEN row_to_json(NEW) ELSE NULL END
        ),
        'clinical_decision_support'
    FROM (SELECT 1) dummy;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_predictions_trigger
    AFTER INSERT OR UPDATE OR DELETE ON predictions
    FOR EACH ROW EXECUTE FUNCTION audit_prediction_changes();

-- ===========================================
-- UTILITY FUNCTIONS
-- ===========================================

-- Function to get current user ID (would integrate with authentication system)
CREATE OR REPLACE FUNCTION current_user_id()
RETURNS VARCHAR(64) AS $$
BEGIN
    -- In production, this would get user ID from session/application context
    RETURN COALESCE(current_setting('app.user_id', TRUE), 'system');
END;
$$ LANGUAGE plpgsql;

-- Function to check data retention compliance
CREATE OR REPLACE FUNCTION check_data_retention()
RETURNS TABLE (
    table_name TEXT,
    old_records BIGINT,
    retention_days INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'patients'::TEXT as table_name,
        COUNT(*) as old_records,
        2555 as retention_days -- 7 years
    FROM patients
    WHERE created_at < CURRENT_DATE - INTERVAL '7 years'

    UNION ALL

    SELECT
        'predictions'::TEXT,
        COUNT(*),
        2555
    FROM predictions
    WHERE created_at < CURRENT_DATE - INTERVAL '7 years';
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- PERFORMANCE OPTIMIZATIONS
-- ===========================================

-- Partitioning for large tables (predictions table by month)
CREATE TABLE predictions_y2024m01 PARTITION OF predictions
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE predictions_y2024m02 PARTITION OF predictions
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Create indexes on partitioned tables
CREATE INDEX idx_predictions_y2024m01_series ON predictions_y2024m01(series_uid);
CREATE INDEX idx_predictions_y2024m01_created ON predictions_y2024m01(created_at);

-- Compression for old partitions
ALTER TABLE predictions_y2024m01 SET (
    autovacuum_enabled = true,
    toast.autovacuum_enabled = true,
    toast.compression = 'pglz'
);