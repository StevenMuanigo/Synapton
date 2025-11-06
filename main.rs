//! Synaptron Server
//!
//! High-performance multi-modal inference engine with dynamic model graph and auto-optimization.

use synaptron::{config::Config, engine::InferenceEngine, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    tracing_subscriber::fmt::init();
    
    // Load configuration
    let config = Config::load()?;
    
    // Create inference engine
    let engine = InferenceEngine::new(config).await?;
    
    // Start server
    engine.start_server().await?;
    
    Ok(())
}
