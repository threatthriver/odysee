use ndarray::{Array1, Array2, Array3, Array4, Axis};
use pyo3::prelude::*;
use rayon::prelude::*;
use image::{ImageBuffer, Rgb};
use std::sync::Arc;

const CHUNK_SIZE: usize = 4096;
const MAX_CONTEXT_LENGTH: usize = 4_000_000;

#[pyclass]
pub struct MultiModalRouter {
    text_embeddings: Array2<f32>,
    image_embeddings: Option<Array4<f32>>,
    routing_dim: usize,
    num_heads: usize,
}

#[pymethods]
impl MultiModalRouter {
    #[new]
    fn new(routing_dim: usize, num_heads: usize) -> Self {
        MultiModalRouter {
            text_embeddings: Array2::zeros((0, routing_dim)),
            image_embeddings: None,
            routing_dim,
            num_heads,
        }
    }

    fn route_text(
        &self,
        queries: Vec<f32>,
        batch_size: usize,
        seq_len: usize,
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let queries = Array2::from_shape_vec((batch_size * seq_len, self.routing_dim), queries)?;
        
        // Process in chunks for memory efficiency
        let chunks: Vec<_> = (0..seq_len).step_by(CHUNK_SIZE).collect();
        let mut all_weights = Vec::new();
        let mut all_indices = Vec::new();

        chunks.par_iter().for_each(|&start| {
            let end = (start + CHUNK_SIZE).min(seq_len);
            let chunk_queries = queries.slice(s![start..end, ..]);
            
            // Compute routing scores
            let scores = chunk_queries.dot(&self.text_embeddings.t());
            
            // Get top-k routes
            let mut scores_vec: Vec<_> = scores.iter().copied().enumerate().collect();
            scores_vec.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let (indices, weights): (Vec<_>, Vec<_>) = scores_vec
                .into_iter()
                .take(self.num_heads)
                .unzip();
            
            // Normalize weights
            let sum: f32 = weights.iter().sum();
            let norm_weights: Vec<f32> = weights.into_iter().map(|w| w / sum).collect();
            
            all_weights.extend(norm_weights);
            all_indices.extend(indices);
        });

        Ok((all_weights, all_indices))
    }

    fn route_image(
        &self,
        image_queries: Vec<f32>,
        image_size: (usize, usize),
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let (height, width) = image_size;
        let queries = Array3::from_shape_vec((height, width, self.routing_dim), image_queries)?;
        
        // Process image patches in parallel
        let patch_size = 16;
        let num_patches_h = (height + patch_size - 1) / patch_size;
        let num_patches_w = (width + patch_size - 1) / patch_size;
        
        let mut all_weights = Vec::new();
        let mut all_indices = Vec::new();
        
        (0..num_patches_h).into_par_iter().for_each(|h| {
            (0..num_patches_w).into_par_iter().for_each(|w| {
                let h_start = h * patch_size;
                let w_start = w * patch_size;
                let h_end = (h_start + patch_size).min(height);
                let w_end = (w_start + patch_size).min(width);
                
                let patch = queries.slice(s![h_start..h_end, w_start..w_end, ..]);
                let patch_mean = patch.mean_axis(Axis(0)).unwrap().mean_axis(Axis(0)).unwrap();
                
                if let Some(ref image_emb) = self.image_embeddings {
                    // Compute patch routing scores
                    let scores = patch_mean.dot(&image_emb.view().into_shape((image_emb.len(), -1)).unwrap().t());
                    
                    // Get top-k routes
                    let mut scores_vec: Vec<_> = scores.iter().copied().enumerate().collect();
                    scores_vec.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    
                    let (indices, weights): (Vec<_>, Vec<_>) = scores_vec
                        .into_iter()
                        .take(self.num_heads)
                        .unzip();
                    
                    // Normalize weights
                    let sum: f32 = weights.iter().sum();
                    let norm_weights: Vec<f32> = weights.into_iter().map(|w| w / sum).collect();
                    
                    all_weights.extend(norm_weights);
                    all_indices.extend(indices);
                }
            });
        });

        Ok((all_weights, all_indices))
    }

    fn update_embeddings(
        &mut self,
        text_embeddings: Option<Vec<f32>>,
        image_embeddings: Option<Vec<f32>>,
        text_shape: Option<(usize, usize)>,
        image_shape: Option<(usize, usize, usize, usize)>,
    ) -> PyResult<()> {
        // Update text embeddings
        if let Some(text_emb) = text_embeddings {
            let (num_tokens, dim) = text_shape.unwrap();
            self.text_embeddings = Array2::from_shape_vec((num_tokens, dim), text_emb)?;
        }
        
        // Update image embeddings
        if let Some(image_emb) = image_embeddings {
            let (batch, channels, height, width) = image_shape.unwrap();
            self.image_embeddings = Some(Array4::from_shape_vec(
                (batch, channels, height, width),
                image_emb,
            )?);
        }
        
        Ok(())
    }
}

// Register the module with Python
#[pymodule]
fn routing(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MultiModalRouter>()?;
    Ok(())
}
