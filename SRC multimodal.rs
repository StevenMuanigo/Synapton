//! Multi-modal input handling for the Synaptron inference engine

use crate::{model::ModelInputType, error::SynaptronError};
use tracing::debug;

/// Multi-modal input processor
pub struct MultimodalProcessor;

impl MultimodalProcessor {
    /// Create a new multi-modal processor
    pub fn new() -> Self {
        Self
    }
    
    /// Detect input type from data
    pub fn detect_input_type(&self, data: &[u8]) -> Result<ModelInputType, SynaptronError> {
        debug!("Detecting input type from data");
        
        // Try to detect input type based on data characteristics
        if self.is_text_data(data) {
            return Ok(ModelInputType::Text);
        }
        
        if self.is_image_data(data) {
            return Ok(ModelInputType::Image);
        }
        
        if self.is_audio_data(data) {
            return Ok(ModelInputType::Audio);
        }
        
        // Default to text if unable to detect
        Ok(ModelInputType::Text)
    }
    
    /// Check if data is text
    fn is_text_data(&self, data: &[u8]) -> bool {
        // Simple heuristic: check if data is valid UTF-8 and doesn't contain null bytes
        if let Ok(text) = std::str::from_utf8(data) {
            // Check if it looks like text (not binary)
            !text.contains('\0') && text.chars().all(|c| c.is_alphabetic() || c.is_numeric() || c.is_whitespace() || c.is_ascii_punctuation())
        } else {
            false
        }
    }
    
    /// Check if data is image
    fn is_image_data(&self, data: &[u8]) -> bool {
        // Check for common image file signatures
        if data.len() < 4 {
            return false;
        }
        
        // JPEG signature
        if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
            return true;
        }
        
        // PNG signature
        if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
            return true;
        }
        
        // GIF signature
        if data[0] == 0x47 && data[1] == 0x49 && data[2] == 0x46 {
            return true;
        }
        
        false
    }
    
    /// Check if data is audio
    fn is_audio_data(&self, data: &[u8]) -> bool {
        // Check for common audio file signatures
        if data.len() < 4 {
            return false;
        }
        
        // WAV signature
        if data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46 {
            return true;
        }
        
        // MP3 signature
        if data.len() >= 3 && data[0] == 0x49 && data[1] == 0x44 && data[2] == 0x33 {
            return true;
        }
        
        // FLAC signature
        if data.len() >= 4 && data[0] == 0x66 && data[1] == 0x4C && data[2] == 0x61 && data[3] == 0x43 {
            return true;
        }
        
        false
    }
    
    /// Route input to appropriate model based on type
    pub async fn route_input(
        &self,
        data: Vec<u8>,
        models: &std::collections::HashMap<String, crate::model::Model>,
    ) -> Result<(Vec<u8>, String), SynaptronError> {
        debug!("Routing input to appropriate model");
        
        // Detect input type
        let input_type = self.detect_input_type(&data)?;
        
        // Find a model that matches the input type
        for (model_name, model) in models {
            if model.input_type == input_type {
                return Ok((data, model_name.clone()));
            }
        }
        
        // If no model matches, return an error
        Err(SynaptronError::Multimodal(
            format!("No model found for input type: {:?}", input_type)
        ))
    }
}
