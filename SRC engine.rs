//! Core inference engine implementation for Synaptron

use crate::{
    config::Config, 
    error::SynaptronError, 
    model::Model, 
    backend::Backend, 
    device::DeviceManager,
    batch::BatchProcessor,
    cache::ModelCache,
    graph::ModelGraph,
    optimizer::AutoOptimizer
};
use tracing::{info, error, debug};
use std::sync::Arc;
use tokio::sync::RwLock;
use axum::{
    routing::{get, post},
    Router,
};

/// Inference Engine
pub struct InferenceEngine {
    /// Configuration
    config: Config,

    /// Active models
    models: Arc<RwLock<std::collections::HashMap<String, Model>>>,

    /// Backend manager
    backends: Arc<RwLock<std::collections::HashMap<String, Box<dyn Backend>>>>,

    /// Device manager
    device_manager: DeviceManager,

    /// Batch processor
    batch_processor: BatchProcessor,

    /// Model cache
    model_cache: ModelCache,

    /// Model graph for chaining
    model_graph: ModelGraph,

    /// Auto optimizer
    auto_optimizer: AutoOptimizer,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub async fn new(config: Config) -> Result<Self, SynaptronError> {
        info!("Initializing Synaptron inference engine");
        
        let device_manager = DeviceManager::new(&config.device);
        let batch_processor = BatchProcessor::new(&config.batch);
        let model_cache = ModelCache::new(&config.cache);
        let model_graph = ModelGraph::new();
        let auto_optimizer = AutoOptimizer::new(&config.backend);
        
        // Create cache directory if it doesn't exist
        tokio::fs::create_dir_all(&config.model.cache_dir).await?;
        
        Ok(Self {
            config,
            models: Arc::new(RwLock::new(std::collections::HashMap::new())),
            backends: Arc::new(RwLock::new(std::collections::HashMap::new())),
            device_manager,
            batch_processor,
            model_cache,
            model_graph,
            auto_optimizer,
        })
    }

    /// Load a model
    pub async fn load_model(&self, model_path: &str) -> Result<(), SynaptronError> {
        info!("Loading model from: {}", model_path);
        
        // Check cache first
        if let Some(cached_model) = self.model_cache.get(model_path).await {
            info!("Model found in cache");
            let mut models_guard = self.models.write().await;
            models_guard.insert(cached_model.name.clone(), cached_model);
            return Ok(());
        }
        
        // Create model instance
        let model = Model::load(model_path, &self.config.model).await?;
        
        // Select optimal device
        let device = self.device_manager.select_device().await?;
        info!("Selected device: {:?}", device);
        
        // Initialize backend
        let backend = self.initialize_backend(&device).await?;
        
        // Optimize model
        let optimized_model = self.auto_optimizer.optimize(model, &device).await?;
        
        // Load model to backend
        backend.load_model(&optimized_model).await?;
        
        // Store model and backend
        {
            let mut models_guard = self.models.write().await;
            models_guard.insert(optimized_model.name.clone(), optimized_model);
        }
        
        {
            let mut backends_guard = self.backends.write().await;
            backends_guard.insert(device.clone(), backend);
        }
        
        info!("Model loaded successfully");
        Ok(())
    }

    /// Initialize backend based on device
    async fn initialize_backend(&self, device: &str) -> Result<Box<dyn Backend>, SynaptronError> {
        match device {
            #[cfg(feature = "openvino")]
            "cpu" | "gpu" | "vpu" => {
                debug!("Initializing OpenVINO backend for device: {}", device);
                Ok(Box::new(crate::backend::openvino::OpenVINOBackend::new(device)?))
            },
            #[cfg(feature = "tensorrt")]
            "cuda" => {
                debug!("Initializing TensorRT backend for CUDA device");
                Ok(Box::new(crate::backend::tensorrt::TensorRTBackend::new()?))
            },
            "cpu" => {
                debug!("Initializing CPU backend");
                Ok(Box::new(crate::backend::cpu::CPUBackend::new()?))
            },
            _ => {
                error!("Unsupported device: {}", device);
                Err(SynaptronError::DeviceSelection(format!("Unsupported device: {}", device)))
            }
        }
    }

    /// Run inference
    pub async fn infer(&self, input: Vec<u8>) -> Result<Vec<u8>, SynaptronError> {
        debug!("Running inference");
        
        // For now, we'll use a simple approach
        // In a real implementation, this would be more complex with model selection, etc.
        
        // Get the first available model
        let models_guard = self.models.read().await;
        let model = models_guard.values().next()
            .ok_or_else(|| SynaptronError::Inference("No model loaded".to_string()))?;
        
        // Get backend
        let backends_guard = self.backends.read().await;
        let backend = backends_guard.values().next()
            .ok_or_else(|| SynaptronError::Inference("No backend available".to_string()))?;
        
        // Run inference
        let result = backend.infer(input).await?;
        
        Ok(result)
    }

    /// Run batch inference
    pub async fn batch_infer(&self, inputs: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, SynaptronError> {
        debug!("Running batch inference with {} inputs", inputs.len());
        
        // Use batch processor
        let results = self.batch_processor.process(inputs, |input| {
            // This is a simplified implementation
            // In a real implementation, this would properly handle async
            Box::pin(async move {
                // For now, we'll just return the input as output
                // In a real implementation, this would call the actual inference
                Ok(input)
            })
        }).await?;
        
        Ok(results)
    }

    /// Start HTTP server
    pub async fn start_server(&self) -> Result<(), SynaptronError> {
        info!("Starting HTTP server on {}:{}", self.config.server.host, self.config.server.port);
        
        // Create router
        let app = self.create_router()?;
        
        // Bind and start server
        let addr = format!("{}:{}", self.config.server.host, self.config.server.port)
            .parse()
            .map_err(|e| SynaptronError::HttpServer(axum::http::Error::from(e)))?;
            
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }

    /// Create HTTP router
    fn create_router(&self) -> Result<Router, SynaptronError> {
        let app = Router::new()
            .route("/predict", post(crate::api::handlers::predict_handler))
            .route("/models", get(crate::api::handlers::list_models_handler))
            .route("/models/activate", post(crate::api::handlers::activate_model_handler))
            .route("/health", get(crate::api::handlers::health_handler))
            .route("/metrics", get(crate::api::handlers::metrics_handler))
            .with_state(self.clone());
            
        Ok(app)
    }
}

impl Clone for InferenceEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            models: self.models.clone(),
            backends: self.backends.clone(),
            device_manager: self.device_manager.clone(),
            batch_processor: self.batch_processor.clone(),
            model_cache: self.model_cache.clone(),
            model_graph: self.model_graph.clone(),
            auto_optimizer: self.auto_optimizer.clone(),
        }
    }
}
