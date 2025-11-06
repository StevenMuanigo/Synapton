//! Error types for the Synaptron inference engine

use thiserror::Error;

/// Synaptron error types
#[derive(Error, Debug)]
pub enum SynaptronError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML serialization error
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    /// HTTP server error
    #[error("HTTP server error: {0}")]
    HttpServer(#[from] axum::http::Error),

    /// Model loading error
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    /// Device selection error
    #[error("Device selection error: {0}")]
    DeviceSelection(String),

    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),

    /// Backend initialization error
    #[error("Backend initialization error: {0}")]
    BackendInit(String),

    /// Tokenization error
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Graph execution error
    #[error("Graph execution error: {0}")]
    GraphExecution(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    Optimization(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// Batch processing error
    #[error("Batch processing error: {0}")]
    Batch(String),

    /// Multi-modal input error
    #[error("Multi-modal input error: {0}")]
    Multimodal(String),

    /// Any other error
    #[error("Other error: {0}")]
    Other(String),
}
