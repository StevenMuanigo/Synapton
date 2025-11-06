//! Metrics and monitoring for the Synaptron inference engine

use crate::error::SynaptronError;
use tracing::info;
use std::sync::atomic::{AtomicU64, AtomicF64, Ordering};
use std::sync::Arc;

/// Metrics collector
pub struct MetricsCollector {
    /// Total number of requests
    total_requests: Arc<AtomicU64>,
    
    /// Total latency in milliseconds
    total_latency_ms: Arc<AtomicF64>,
    
    /// Total number of successful requests
    successful_requests: Arc<AtomicU64>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            total_requests: Arc::new(AtomicU64::new(0)),
            total_latency_ms: Arc::new(AtomicF64::new(0.0)),
            successful_requests: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Record a request
    pub fn record_request(&self, latency_ms: f64, success: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
        
        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get total requests
    pub fn get_total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }
    
    /// Get average latency
    pub fn get_avg_latency_ms(&self) -> f64 {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        
        if total_requests > 0 {
            total_latency / (total_requests as f64)
        } else {
            0.0
        }
    }
    
    /// Get success rate
    pub fn get_success_rate(&self) -> f64 {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let successful_requests = self.successful_requests.load(Ordering::Relaxed);
        
        if total_requests > 0 {
            (successful_requests as f64) / (total_requests as f64) * 100.0
        } else {
            0.0
        }
    }
    
    /// Get throughput (requests per second)
    pub fn get_throughput(&self, uptime_seconds: f64) -> f64 {
        if uptime_seconds > 0.0 {
            (self.total_requests.load(Ordering::Relaxed) as f64) / uptime_seconds
        } else {
            0.0
        }
    }
    
    /// Reset metrics
    pub fn reset(&self) {
        info!("Resetting metrics");
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_latency_ms.store(0.0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            total_requests: self.total_requests.clone(),
            total_latency_ms: self.total_latency_ms.clone(),
            successful_requests: self.successful_requests.clone(),
        }
    }
}
