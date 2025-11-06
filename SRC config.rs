//! Configuration management for the Synaptron inference engine

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use config::{Config as ConfigLoader, Environment, File};

use crate::error::SynaptronError;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address
    pub host: String,

    /// Port number
    pub port: u16,

    /// Number of worker threads
    pub workers: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: num_cpus::get(),
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model cache directory
    pub cache_dir: String,

    /// Default model name
    pub default_model: String,

    /// Maximum input length
    pub max_input_length: usize,

    /// Enable auto-download
    pub auto_download: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            cache_dir: "./models_cache".to_string(),
            default_model: "bert-base-uncased".to_string(),
            max_input_length: 512,
            auto_download: true,
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Preferred device
    pub preferred: String,

    /// Enable auto device selection
    pub auto_select: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            preferred: "cpu".to_string(),
            auto_select: true,
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Enable OpenVINO backend
    pub openvino: bool,

    /// Enable TensorRT backend
    pub tensorrt: bool,

    /// Enable ONNX Runtime backend
    pub onnx_runtime: bool,

    /// Enable auto backend selection
    pub auto_select: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            openvino: false,
            tensorrt: false,
            onnx_runtime: true,
            auto_select: true,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable LRU cache
    pub enabled: bool,

    /// Maximum cache size
    pub max_size: usize,

    /// Cache TTL in seconds
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1000,
            ttl_seconds: 3600,
        }
    }
}

/// Batch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable batching
    pub enabled: bool,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Batch timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_size: 32,
            timeout_ms: 100,
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable tracing
    pub tracing: bool,

    /// Enable metrics
    pub metrics: bool,

    /// Metrics endpoint
    pub metrics_endpoint: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            tracing: true,
            metrics: true,
            metrics_endpoint: "/metrics".to_string(),
        }
    }
}

/// Main configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,

    /// Model configuration
    pub model: ModelConfig,

    /// Device configuration
    pub device: DeviceConfig,

    /// Backend configuration
    pub backend: BackendConfig,

    /// Cache configuration
    pub cache: CacheConfig,

    /// Batch configuration
    pub batch: BatchConfig,

    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            device: DeviceConfig::default(),
            backend: BackendConfig::default(),
            cache: CacheConfig::default(),
            batch: BatchConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from file and environment variables
    pub fn load() -> Result<Self, SynaptronError> {
        let mut config_builder = ConfigLoader::builder()
            .set_default("server.host", "127.0.0.1")?
            .set_default("server.port", 8080)?
            .set_default("server.workers", num_cpus::get())?
            .set_default("model.cache_dir", "./models_cache")?
            .set_default("model.default_model", "bert-base-uncased")?
            .set_default("model.max_input_length", 512)?
            .set_default("model.auto_download", true)?
            .set_default("device.preferred", "cpu")?
            .set_default("device.auto_select", true)?
            .set_default("backend.openvino", false)?
            .set_default("backend.tensorrt", false)?
            .set_default("backend.onnx_runtime", true)?
            .set_default("backend.auto_select", true)?
            .set_default("cache.enabled", true)?
            .set_default("cache.max_size", 1000)?
            .set_default("cache.ttl_seconds", 3600)?
            .set_default("batch.enabled", true)?
            .set_default("batch.max_batch_size", 32)?
            .set_default("batch.timeout_ms", 100)?
            .set_default("monitoring.tracing", true)?
            .set_default("monitoring.metrics", true)?
            .set_default("monitoring.metrics_endpoint", "/metrics")?
            .add_source(Environment::with_prefix("SYNAPTRON"));

        // Try to load from config file
        if let Ok(current_dir) = env::current_dir() {
            let config_path = current_dir.join("config.yaml");
            if config_path.exists() {
                config_builder = config_builder.add_source(File::from(config_path));
            }
        }

        let config = config_builder.build()?;
        let synaptron_config: Config = config.try_deserialize()?;

        Ok(synaptron_config)
    }

    /// Save configuration to file
    pub fn save(&self, path: &str) -> Result<(), SynaptronError> {
        let yaml = serde_yaml::to_string(self)?;
        fs::write(path, yaml)?;
        Ok(())
    }
}
