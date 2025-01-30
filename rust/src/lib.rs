use ndarray::{s, Array2, Array3, Array4, Axis};
use pyo3::prelude::*;
use rayon::prelude::*;

// Custom error wrapper
#[derive(Debug)]
struct OdyseeError(String);

impl From<ndarray::ShapeError> for OdyseeError {
    fn from(err: ndarray::ShapeError) -> Self {
        OdyseeError(format!("Shape error: {}", err))
    }
}

impl From<OdyseeError> for PyErr {
    fn from(err: OdyseeError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.0)
    }
}

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
        let queries = Array2::from_shape_vec((batch_size * seq_len, self.routing_dim), queries)
            .map_err(OdyseeError::from)?;
        
        // Process in chunks for memory efficiency
        let chunks: Vec<_> = (0..seq_len).step_by(4096).collect();
        
        // Process chunks in parallel and collect results
        let results: Vec<_> = chunks.par_iter().map(|&start| {
            let end = (start + 4096).min(seq_len);
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
            
            (norm_weights, indices)
        }).collect();
        
        // Combine results
        let (all_weights, all_indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let all_weights = all_weights.into_iter().flatten().collect();
        let all_indices = all_indices.into_iter().flatten().collect();

        Ok((all_weights, all_indices))
    }

    fn route_image(
        &self,
        image_queries: Vec<f32>,
        image_size: (usize, usize),
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let (height, width) = image_size;
        let queries = Array3::from_shape_vec((height, width, self.routing_dim), image_queries)
            .map_err(OdyseeError::from)?;
        
        // Process image patches in parallel
        let patch_size = 16;
        let num_patches_h = (height + patch_size - 1) / patch_size;
        let num_patches_w = (width + patch_size - 1) / patch_size;
        
        // Create patch coordinates
        let patches: Vec<_> = (0..num_patches_h)
            .flat_map(|h| (0..num_patches_w).map(move |w| (h, w)))
            .collect();
        
        // Process patches in parallel and collect results
        let results: Vec<_> = patches.par_iter().map(|&(h, w)| {
            let h_start = h * patch_size;
            let w_start = w * patch_size;
            let h_end = (h_start + patch_size).min(height);
            let w_end = (w_start + patch_size).min(width);
            
            let patch = queries.slice(s![h_start..h_end, w_start..w_end, ..]);
            let patch_mean = patch.mean_axis(Axis(0)).unwrap().mean_axis(Axis(0)).unwrap();
            
            if let Some(ref image_emb) = self.image_embeddings {
                // Compute patch routing scores
                let flat_image_emb = image_emb.view().into_shape((image_emb.len(), self.routing_dim)).unwrap();
                let scores = patch_mean.dot(&flat_image_emb.t());
                
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
                
                Some((norm_weights, indices))
            } else {
                None
            }
        }).collect();
        
        // Combine results
        let (all_weights, all_indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .filter_map(|x| x)
            .unzip();
        let all_weights = all_weights.into_iter().flatten().collect();
        let all_indices = all_indices.into_iter().flatten().collect();

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
            self.text_embeddings = Array2::from_shape_vec((num_tokens, dim), text_emb)
                .map_err(OdyseeError::from)?;
        }
        
        // Update image embeddings
        if let Some(image_emb) = image_embeddings {
            let (batch, channels, height, width) = image_shape.unwrap();
            self.image_embeddings = Some(Array4::from_shape_vec(
                (batch, channels, height, width),
                image_emb,
            ).map_err(OdyseeError::from)?);
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
