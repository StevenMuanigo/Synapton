//! API handlers for the Synaptron inference engine

use crate::engine::InferenceEngine;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    debug_handler,
};
use serde::{Deserialize, Serialize};
use tracing::{info, error};
use std::time::Instant;

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime: u64,
}

/// Predict request
#[derive(Deserialize)]
pub struct PredictRequest {
    pub input: String,
}

/// Predict response
#[derive(Serialize)]
pub struct PredictResponse {
    pub prediction: String,
    pub latency_ms: u128,
}

/// List models response
#[derive(Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<String>,
}

/// Activate model request
#[derive(Deserialize)]
pub struct ActivateModelRequest {
    pub model_name: String,
}

/// Metrics response
#[derive(Serialize)]
pub struct MetricsResponse {
    pub total_requests: u64,
    pub avg_latency_ms: f64,
    pub throughput: f64,
}

/// Health check handler
#[debug_handler]
pub async fn health_handler() -> Result<Json<HealthResponse>, StatusCode> {
    info!("Health check requested");
    
    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime: 0, // In a real implementation, this would track uptime
    };
    
    Ok(Json(response))
}

/// Predict handler
#[debug_handler]
pub async fn predict_handler(
    State(engine): State<InferenceEngine>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    info!("Predict requested for input: {}", &payload.input);
    
    // Start timing
    let start_time = Instant::now();
    
    // Convert input to bytes for processing
    let input_bytes = payload.input.as_bytes().to_vec();
    
    // Run inference
    match engine.infer(input_bytes).await {
        Ok(output_bytes) => {
            // Convert output bytes back to string
            let prediction = String::from_utf8_lossy(&output_bytes).to_string();
            
            // Calculate latency
            let latency_ms = start_time.elapsed().as_millis();
            
            info!("Prediction completed successfully in {} ms", latency_ms);
            
            let response = PredictResponse {
                prediction,
                latency_ms,
            };
            
            Ok(Json(response))
        }
        Err(e) => {
            error!("Prediction failed: {:?}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Prediction failed: {:?}", e)))
        }
    }
}

/// List models handler
#[debug_handler]
pub async fn list_models_handler(
    State(engine): State<InferenceEngine>,
) -> Result<Json<ListModelsResponse>, (StatusCode, String)> {
    info!("List models requested");
    
    // Get list of loaded models
    let models_guard = engine.models.read().await;
    let model_names: Vec<String> = models_guard.keys().cloned().collect();
    
    let response = ListModelsResponse {
        models: model_names,
    };
    
    Ok(Json(response))
}

/// Activate model handler
#[debug_handler]
pub async fn activate_model_handler(
    State(engine): State<InferenceEngine>,
    Json(payload): Json<ActivateModelRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    info!("Activate model requested: {}", payload.model_name);
    
    // In a real implementation, this would load/activate the specified model
    // For now, we'll just return OK
    Ok(StatusCode::OK)
}

/// Metrics handler
#[debug_handler]
pub async fn metrics_handler(
    State(_engine): State<InferenceEngine>,
) -> Result<Json<MetricsResponse>, (StatusCode, String)> {
    info!("Metrics requested");
    
    // In a real implementation, this would return actual metrics
    let response = MetricsResponse {
        total_requests: 0,
        avg_latency_ms: 0.0,
        throughput: 0.0,
    };
    
    Ok(Json(response))
}
