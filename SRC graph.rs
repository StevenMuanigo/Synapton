//! Dynamic model graph implementation for the Synaptron inference engine

use crate::{model::Model, error::SynaptronError};
use tracing::{info, debug};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Node ID
    pub id: String,
    
    /// Model name
    pub model_name: String,
    
    /// Input node IDs
    pub inputs: Vec<String>,
    
    /// Output node IDs
    pub outputs: Vec<String>,
}

/// Model graph
pub struct ModelGraph {
    /// Graph nodes
    nodes: HashMap<String, GraphNode>,
    
    /// Execution order
    execution_order: Vec<String>,
}

impl ModelGraph {
    /// Create a new model graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            execution_order: Vec::new(),
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), SynaptronError> {
        info!("Adding node to graph: {}", node.id);
        
        self.nodes.insert(node.id.clone(), node);
        self.update_execution_order()?;
        
        Ok(())
    }
    
    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), SynaptronError> {
        info!("Removing node from graph: {}", node_id);
        
        self.nodes.remove(node_id);
        self.update_execution_order()?;
        
        Ok(())
    }
    
    /// Update execution order based on dependencies
    fn update_execution_order(&mut self) -> Result<(), SynaptronError> {
        debug!("Updating execution order");
        
        // Simple topological sort implementation
        let mut visited = HashMap::new();
        let mut order = Vec::new();
        
        for (node_id, _) in &self.nodes {
            if !visited.contains_key(node_id) {
                self.topological_sort(node_id, &mut visited, &mut order)?;
            }
        }
        
        self.execution_order = order;
        Ok(())
    }
    
    /// Topological sort helper
    fn topological_sort(
        &self,
        node_id: &str,
        visited: &mut HashMap<String, bool>,
        order: &mut Vec<String>,
    ) -> Result<(), SynaptronError> {
        // Mark as visited
        visited.insert(node_id.to_string(), true);
        
        // Visit all dependencies first
        if let Some(node) = self.nodes.get(node_id) {
            for input_id in &node.inputs {
                if self.nodes.contains_key(input_id) && !visited.contains_key(input_id) {
                    self.topological_sort(input_id, visited, order)?;
                }
            }
        }
        
        // Add to order
        order.push(node_id.to_string());
        Ok(())
    }
    
    /// Execute the graph
    pub async fn execute(
        &self,
        models: &HashMap<String, Model>,
        initial_input: Vec<u8>,
    ) -> Result<Vec<u8>, SynaptronError> {
        info!("Executing model graph");
        
        let mut outputs: HashMap<String, Vec<u8>> = HashMap::new();
        outputs.insert("input".to_string(), initial_input);
        
        // Execute nodes in order
        for node_id in &self.execution_order {
            if let Some(node) = self.nodes.get(node_id) {
                // Collect inputs for this node
                let mut node_inputs = Vec::new();
                
                if node.inputs.is_empty() {
                    // Use initial input if no specific inputs
                    node_inputs.push(outputs.get("input").unwrap().clone());
                } else {
                    // Collect from previous node outputs
                    for input_id in &node.inputs {
                        if let Some(output) = outputs.get(input_id) {
                            node_inputs.push(output.clone());
                        }
                    }
                }
                
                // For simplicity, we'll use the first input
                // In a real implementation, this would be more complex
                let input = if !node_inputs.is_empty() {
                    node_inputs[0].clone()
                } else {
                    return Err(SynaptronError::GraphExecution(
                        format!("No input available for node: {}", node_id)
                    ));
                };
                
                // Run inference with the model
                if let Some(model) = models.get(&node.model_name) {
                    // In a real implementation, this would run the actual inference
                    // For now, we'll just pass the input through
                    let output = input;
                    outputs.insert(node_id.clone(), output);
                } else {
                    return Err(SynaptronError::GraphExecution(
                        format!("Model not found: {}", node.model_name)
                    ));
                }
            }
        }
        
        // Return the output of the last node
        if let Some(last_node_id) = self.execution_order.last() {
            if let Some(output) = outputs.get(last_node_id) {
                Ok(output.clone())
            } else {
                Err(SynaptronError::GraphExecution(
                    "No output from graph execution".to_string()
                ))
            }
        } else {
            // Return initial input if no nodes
            Ok(outputs.get("input").unwrap().clone())
        }
    }
}

impl Clone for ModelGraph {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            execution_order: self.execution_order.clone(),
        }
    }
}
