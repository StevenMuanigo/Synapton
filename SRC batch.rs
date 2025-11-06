/! Batch processing implementation for the Synaptron inference engine

use crate::{config::BatchConfig, error::SynaptronError};
use tracing::{info, debug};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};

/// Batch processor
pub struct BatchProcessor {
    /// Batch configuration
    config: BatchConfig,
    
    /// Current batch
    current_batch: Arc<RwLock<Vec<Vec<u8>>>>,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: &BatchConfig) -> Self {
        Self {
            config: config.clone(),
            current_batch: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Process inputs in batches
    pub async fn process<F, Fut>(
        &self,
        inputs: Vec<Vec<u8>>,
        processor: F,
    ) -> Result<Vec<Vec<u8>>, SynaptronError>
    where
        F: Fn(Vec<u8>) -> Fut,
        Fut: Future<Output = Result<Vec<u8>, SynaptronError>>,
    {
        if !self.config.enabled || inputs.len() < 2 {
            // Process individually if batching is disabled or only one input
            debug!("Processing inputs individually");
            let mut results = Vec::new();
            
            for input in inputs {
                let result = processor(input).await?;
                results.push(result);
            }
            
            return Ok(results);
        }
        
        debug!("Processing batch of {} inputs", inputs.len());
        
        // Split inputs into batches
        let mut results = Vec::new();
        
        for chunk in inputs.chunks(self.config.max_batch_size) {
            let batch_results = self.process_batch(chunk.to_vec(), &processor).await?;
            results.extend(batch_results);
        }
        
        Ok(results)
    }
    
    /// Process a single batch
    async fn process_batch<F, Fut>(
        &self,
        batch: Vec<Vec<u8>>,
        processor: &F,
    ) -> Result<Vec<Vec<u8>>, SynaptronError>
    where
        F: Fn(Vec<u8>) -> Fut,
        Fut: Future<Output = Result<Vec<u8>, SynaptronError>>,
    {
        info!("Processing batch of size {}", batch.len());
        
        // Process all inputs in the batch concurrently
        let mut futures = Vec::new();
        
        for input in batch {
            futures.push(processor(input));
        }
        
        // Wait for all results with timeout
        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
        
        let results = match timeout(timeout_duration, futures::future::join_all(futures)).await {
            Ok(results) => results,
            Err(_) => {
                return Err(SynaptronError::Batch(
                    "Batch processing timed out".to_string()
                ));
            }
        };
        
        // Collect results
        let mut processed_results = Vec::new();
        
        for result in results {
            processed_results.push(result?);
        }
        
        Ok(processed_results)
    }
}
