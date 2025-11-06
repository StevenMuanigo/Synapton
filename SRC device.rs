//! Device management for the Synaptron inference engine

use crate::{config::DeviceConfig, error::SynaptronError};
use tracing::{info, debug};

/// Device manager
pub struct DeviceManager {
    /// Device configuration
    config: DeviceConfig,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new(config: &DeviceConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
    
    /// Select the best device for inference
    pub async fn select_device(&self) -> Result<String, SynaptronError> {
        info!("Selecting best device for inference");
        
        if self.config.auto_select {
            // In a real implementation, this would check available hardware
            // and select the best device based on performance characteristics
            
            // For now, we'll use a simple selection:
            #[cfg(feature = "cuda")]
            {
                // Check if CUDA is available
                if self.is_cuda_available().await {
                    info!("Selected CUDA device");
                    return Ok("cuda".to_string());
                }
            }
            
            #[cfg(feature = "openvino")]
            {
                // Check if Intel hardware is available
                if self.is_intel_hardware_available().await {
                    info!("Selected Intel GPU/VPU device");
                    return Ok("gpu".to_string()); // or "vpu"
                }
            }
            
            // Default to CPU
            info!("Selected CPU device");
            Ok("cpu".to_string())
        } else {
            // Use preferred device from config
            info!("Using preferred device: {}", self.config.preferred);
            Ok(self.config.preferred.clone())
        }
    }
    
    /// Check if CUDA is available
    async fn is_cuda_available(&self) -> bool {
        debug!("Checking CUDA availability");
        
        // In a real implementation, this would check for CUDA devices
        // For now, we'll return false
        false
    }
    
    /// Check if Intel hardware is available
    async fn is_intel_hardware_available(&self) -> bool {
        debug!("Checking Intel hardware availability");
        
        // In a real implementation, this would check for Intel devices
        // For now, we'll return false
        false
    }
}

impl Clone for DeviceManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}//! Device management for the Synaptron inference engine

use crate::{config::DeviceConfig, error::SynaptronError};
use tracing::{info, debug};

/// Device manager
pub struct DeviceManager {
    /// Device configuration
    config: DeviceConfig,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new(config: &DeviceConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
    
    /// Select the best device for inference
    pub async fn select_device(&self) -> Result<String, SynaptronError> {
        info!("Selecting best device for inference");
        
        if self.config.auto_select {
            // In a real implementation, this would check available hardware
            // and select the best device based on performance characteristics
            
            // For now, we'll use a simple selection:
            #[cfg(feature = "cuda")]
            {
                // Check if CUDA is available
                if self.is_cuda_available().await {
                    info!("Selected CUDA device");
                    return Ok("cuda".to_string());
                }
            }
            
            #[cfg(feature = "openvino")]
            {
                // Check if Intel hardware is available
                if self.is_intel_hardware_available().await {
                    info!("Selected Intel GPU/VPU device");
                    return Ok("gpu".to_string()); // or "vpu"
                }
            }
            
            // Default to CPU
            info!("Selected CPU device");
            Ok("cpu".to_string())
        } else {
            // Use preferred device from config
            info!("Using preferred device: {}", self.config.preferred);
            Ok(self.config.preferred.clone())
        }
    }
    
    /// Check if CUDA is available
    async fn is_cuda_available(&self) -> bool {
        debug!("Checking CUDA availability");
        
        // In a real implementation, this would check for CUDA devices
        // For now, we'll return false
        false
    }
    
    /// Check if Intel hardware is available
    async fn is_intel_hardware_available(&self) -> bool {
        debug!("Checking Intel hardware availability");
        
        // In a real implementation, this would check for Intel devices
        // For now, we'll return false
        false
    }
}

impl Clone for DeviceManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
        }
    }
}
