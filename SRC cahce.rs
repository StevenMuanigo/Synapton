//! Model cache implementation for the Synaptron inference engine

use crate::{config::CacheConfig, model::Model, error::SynaptronError};
use tracing::{info, debug};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cached model entry
struct CachedModel {
    /// The model
    model: Model,
    
    /// Timestamp when cached
    timestamp: u64,
    
    /// Access count
    access_count: usize,
}

/// Model Cache
pub struct ModelCache {
    /// Cache configuration
    config: CacheConfig,
    
    /// Cached models
    cache: Arc<RwLock<HashMap<String, CachedModel>>>,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(config: &CacheConfig) -> Self {
        Self {
            config: config.clone(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Get a model from cache
    pub async fn get(&self, model_path: &str) -> Option<Model> {
        if !self.config.enabled {
            return None;
        }
        
        debug!("Checking cache for model: {}", model_path);
        
        let mut cache_guard = self.cache.write().await;
        
        // Check if model exists in cache
        if let Some(mut cached_model) = cache_guard.get_mut(model_path) {
            // Check if cache entry is still valid
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
                
            if current_time - cached_model.timestamp < self.config.ttl_seconds {
                // Update access count
                cached_model.access_count += 1;
                info!("Model found in cache: {}", model_path);
                return Some(cached_model.model.clone());
            } else {
                // Remove expired entry
                cache_guard.remove(model_path);
                info!("Expired model removed from cache: {}", model_path);
            }
        }
        
        None
    }
    
    /// Put a model in cache
    pub async fn put(&self, model: Model) -> Result<(), SynaptronError> {
        if !self.config.enabled {
            return Ok(());
        }
        
        debug!("Putting model in cache: {}", model.path);
        
        let mut cache_guard = self.cache.write().await;
        
        // Check cache size and evict if necessary
        if cache_guard.len() >= self.config.max_size {
            self.evict_lru(&mut cache_guard).await;
        }
        
        // Add model to cache
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        cache_guard.insert(
            model.path.clone(),
            CachedModel {
                model: model.clone(),
                timestamp,
                access_count: 1,
            },
        );
        
        info!("Model cached: {}", model.path);
        Ok(())
    }
    
    /// Evict least recently used model from cache
    async fn evict_lru(&self, cache: &mut HashMap<String, CachedModel>) {
        if let Some((key, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
        {
            let key = key.clone();
            cache.remove(&key);
            info!("Evicted LRU model from cache: {}", key);
        }
    }
    
    /// Clear cache
    pub async fn clear(&self) -> Result<(), SynaptronError> {
        debug!("Clearing model cache");
        
        let mut cache_guard = self.cache.write().await;
        cache_guard.clear();
        
        info!("Model cache cleared");
        Ok(())
    }
}
