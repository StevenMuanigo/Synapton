//! # Synaptron
//! 
//! High-performance multi-modal inference engine with dynamic model graph and auto-optimization.
//! 
//! ## Features
//! 
//! - Multi-Modal Pipeline: Text, image and audio input support
//! - Dynamic Model Graph: Model chaining with config-based pipeline
//! - Auto-Optimization Layer: Hardware acceleration with backend auto-selection
//! - Memory & Performance: LRU cache and async batching
//! - Advanced API & Dashboard: Prometheus metrics and optional web UI

/// Core inference engine module
pub mod engine;

/// Model management and loading
pub mod model;

/// Hardware acceleration backends
pub mod backend;

/// Memory management and optimization
pub mod memory;

/// Batch processing and streaming
pub mod batch;

/// Quantization and optimization
pub mod quantization;

/// Device selection and management
pub mod device;

/// Error handling
pub mod error;

/// Configuration management
pub mod config;

/// Metrics and monitoring
pub mod metrics;

/// Utilities and helpers
pub mod utils;

/// Command line interface
pub mod cli;

/// API modules
pub mod api;

/// Model cache
pub mod cache;

/// Preprocessing utilities
pub mod preprocessing;

/// Postprocessing utilities
pub mod postprocessing;

/// Dynamic model graph
pub mod graph;

/// Auto-optimization layer
pub mod optimizer;

/// Multi-modal input handling
pub mod multimodal;

// Re-export main types
pub use engine::InferenceEngine;
pub use model::Model;
pub use config::Config;
pub use error::SynaptronError;

/// Result type
pub type Result<T> = std::result::Result<T, SynaptronError>;
