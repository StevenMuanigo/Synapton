/! Model management and loading for the Synaptron inference engine

use crate::{config::ModelConfig, error::SynaptronError};
use tracing::{info, debug, warn};
use std::path::Path;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;

/// Model input types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelInputType {
    Text,
    Image,
    Audio,
}

/// Model representation
pub struct Model {
    /// Model name
    pub name: String,

    /// Model path
    pub path: String,

    /// Model format
    pub format: String,

    /// Model input type
    pub input_type: ModelInputType,

    /// Model metadata
    pub metadata: ModelMetadata,

    /// Loaded model data
    pub data: Vec<u8>,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Input dimensions
    pub input_shape: Vec<usize>,

    /// Output dimensions
    pub output_shape: Vec<usize>,

    /// Data type
    pub data_type: String,

    /// Model size in bytes
    pub size: usize,

    /// Model architecture
    pub architecture: String,

    /// Model version
    pub version: String,

    /// Required libraries
    pub required_libs: Vec<String>,
}

impl Model {
    /// Load model from file
    pub async fn load(path: &str, config: &ModelConfig) -> Result<Self, SynaptronError> {
        info!("Loading model from: {}", path);
        
        // Check if file exists
        if !Path::new(path).exists() {
            // Try to download from Hugging Face if auto-download is enabled
            if config.auto_download {
                info!("Model not found locally, attempting to download from Hugging Face");
                Self::download_from_huggingface(path, config).await?;
            } else {
                return Err(SynaptronError::ModelLoad(format!("Model file not found: {}", path)));
            }
        }
        
        // Read model data
        let data = fs::read(path).await?;
        let size = data.len();
        
        // Get model name from path
        let name = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Determine model format
        let format = Self::detect_format(path)?;
        
        // Determine input type
        let input_type = Self::detect_input_type(&name, &format)?;
        
        // Extract metadata from config.json if available
        let metadata = Self::extract_metadata(path).await?;
        
        info!("Model loaded successfully. Size: {} bytes, Format: {}, Input Type: {:?}", size, format, input_type);
        
        Ok(Self {
            name,
            path: path.to_string(),
            format,
            input_type,
            metadata,
            data,
        })
    }

    /// Detect model format from file extension
    fn detect_format(path: &str) -> Result<String, SynaptronError> {
        let path = Path::new(path);
        let extension = path.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        match extension.as_str() {
            "onnx" => Ok("onnx".to_string()),
            "pt" | "pth" => Ok("pytorch".to_string()),
            "pb" => Ok("savedmodel".to_string()),
            "ts" => Ok("torchscript".to_string()),
            "gguf" => Ok("gguf".to_string()),
            "safetensors" => Ok("safetensors".to_string()),
            _ => Ok("unknown".to_string()),
        }
    }

    /// Detect input type based on model name and format
    fn detect_input_type(name: &str, format: &str) -> Result<ModelInputType, SynaptronError> {
        let name_lower = name.to_lowercase();
        
        // Text models
        if name_lower.contains("bert") || 
           name_lower.contains("roberta") || 
           name_lower.contains("distilbert") || 
           name_lower.contains("gpt") ||
           name_lower.contains("llama") ||
           name_lower.contains("mistral") ||
           name_lower.contains("t5") ||
           name_lower.contains("bart") {
            return Ok(ModelInputType::Text);
        }
        
        // Image models
        if name_lower.contains("clip") || 
           name_lower.contains("resnet") || 
           name_lower.contains("vit") || 
           name_lower.contains("efficientnet") ||
           name_lower.contains("mobilenet") ||
           name_lower.contains("yolo") {
            return Ok(ModelInputType::Image);
        }
        
        // Audio models
        if name_lower.contains("whisper") || 
           name_lower.contains("wav2vec") || 
           name_lower.contains("hubert") || 
           name_lower.contains("speecht5") {
            return Ok(ModelInputType::Audio);
        }
        
        // Default to text for ONNX models (most common)
        if format == "onnx" {
            warn!("Could not determine input type, defaulting to text for ONNX model");
            return Ok(ModelInputType::Text);
        }
        
        // Try to determine from format
        match format {
            "onnx" | "pytorch" | "torchscript" | "savedmodel" => Ok(ModelInputType::Text),
            _ => Ok(ModelInputType::Text), // Default fallback
        }
    }

    /// Extract metadata from config.json
    async fn extract_metadata(path: &str) -> Result<ModelMetadata, SynaptronError> {
        let model_dir = Path::new(path).parent().unwrap_or(Path::new("."));
        let config_path = model_dir.join("config.json");
        
        if config_path.exists() {
            let config_data = fs::read_to_string(&config_path).await?;
            let config: HashMap<String, serde_json::Value> = serde_json::from_str(&config_data)?;
            
            // Extract common metadata fields
            let architecture = config.get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            
            let version = config.get("version")
                .and_then(|v| v.as_str())
                .unwrap_or("1.0")
                .to_string();
            
            // Create metadata
            Ok(ModelMetadata {
                input_shape: vec![1, 3, 224, 224], // Default values
                output_shape: vec![1, 1000],
                data_type: "f32".to_string(),
                size: 0, // Will be set when loading
                architecture,
                version,
                required_libs: vec![], // Will be populated based on format
            })
        } else {
            // Default metadata
            Ok(ModelMetadata {
                input_shape: vec![1, 3, 224, 224],
                output_shape: vec![1, 1000],
                data_type: "f32".to_string(),
                size: 0,
                architecture: "unknown".to_string(),
                version: "1.0".to_string(),
                required_libs: vec![],
            })
        }
    }

    /// Download model from Hugging Face
    async fn download_from_huggingface(path: &str, config: &ModelConfig) -> Result<(), SynaptronError> {
        // This is a simplified implementation
        // In a real implementation, this would use the Hugging Face API
        info!("Downloading model to: {}", path);
        
        // Create cache directory if it doesn't exist
        let cache_dir = Path::new(&config.cache_dir);
        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir).await?;
        }
        
        // For now, we'll just create an empty file to simulate download
        // In a real implementation, this would download the actual model
        fs::write(path, Vec::<u8>::new()).await?;
        
        Ok(())
    }

    /// Save model to cache
    pub async fn save_to_cache(&self, cache_dir: &str) -> Result<(), SynaptronError> {
        debug!("Saving model to cache: {}", cache_dir);
        
        let cache_path = format!("{}/{}.cache", cache_dir, self.name);
        fs::write(&cache_path, &self.data).await?;
        
        info!("Model cached to: {}", cache_path);
        Ok(())
    }

    /// Load model from cache
    pub async fn load_from_cache(cache_path: &str) -> Result<Self, SynaptronError> {
        info!("Loading model from cache: {}", cache_path);
        
        let data = fs::read(cache_path).await?;
        let size = data.len();
        
        let name = Path::new(cache_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        let metadata = ModelMetadata {
            input_shape: vec![1, 3, 224, 224],
            output_shape: vec![1, 1000],
            data_type: "f32".to_string(),
            size,
            architecture: "cached".to_string(),
            version: "1.0".to_string(),
            required_libs: vec![],
        };
        
        Ok(Self {
            name,
            path: cache_path.to_string(),
            format: "cached".to_string(),
            input_type: ModelInputType::Text, // Default for cached models
            metadata,
            data,
        })
    }
}
