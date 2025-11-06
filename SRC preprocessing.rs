//! Preprocessing utilities for the Synaptron inference engine

use crate::error::SynaptronError;
use tokenizers::Tokenizer;
use tracing::debug;
use unicode_normalization::UnicodeNormalization;

/// Preprocessing utilities
pub struct Preprocessor {
    /// Text tokenizer
    tokenizer: Option<Tokenizer>,
    
    /// Maximum input length
    max_length: usize,
}

impl Preprocessor {
    /// Create a new preprocessor
    pub fn new(max_length: usize) -> Self {
        Self {
            tokenizer: None,
            max_length,
        }
    }
    
    /// Set tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }
    
    /// Clean text input
    pub fn clean_text(&self, text: &str) -> String {
        debug!("Cleaning text input");
        
        // Normalize unicode
        let normalized = text.nfkc().collect::<String>();
        
        // Remove extra whitespace
        let cleaned = normalized.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        // Truncate to max length
        if cleaned.len() > self.max_length {
            cleaned[..self.max_length].to_string()
        } else {
            cleaned
        }
    }
    
    /// Tokenize text
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, SynaptronError> {
        debug!("Tokenizing text");
        
        match &self.tokenizer {
            Some(tokenizer) => {
                let encoding = tokenizer.encode(text, false)
                    .map_err(|e| SynaptronError::Tokenization(e.to_string()))?;
                Ok(encoding.get_ids().to_vec())
            }
            None => {
                // Fallback to simple character-based tokenization
                Ok(text.chars().map(|c| c as u32).collect())
            }
        }
    }
    
    /// Preprocess text input
    pub fn preprocess_text(&self, text: &str) -> Result<Vec<u32>, SynaptronError> {
        debug!("Preprocessing text input");
        
        let cleaned = self.clean_text(text);
        self.tokenize(&cleaned)
    }
}
