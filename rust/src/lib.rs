use pyo3::prelude::*;
use ndarray::{Array2, Array3};
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

#[pymodule]
fn odysee_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<MultiModalRouter>()?;
    Ok(())
}

#[pyclass]
pub struct MultiModalRouter {
    routing_dim: usize,
    num_heads: usize,
}

#[pymethods]
impl MultiModalRouter {
    #[new]
    fn new(routing_dim: usize, num_heads: usize) -> Self {
        MultiModalRouter {
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            
        let weights: Vec<f32> = (0..batch_size * seq_len)
            .into_par_iter()
            .flat_map(|i| {
                let query = queries.slice(ndarray::s![i, ..]);
                let mut scores = vec![0.0; self.num_heads];
                for h in 0..self.num_heads {
                    scores[h] = query.iter().sum::<f32>() / (self.routing_dim as f32);
                }
                scores
            })
            .collect();
            
        let indices: Vec<usize> = (0..batch_size * seq_len)
            .into_par_iter()
            .flat_map(|i| {
                let start = i * self.num_heads;
                let mut head_indices: Vec<usize> = (0..self.num_heads).collect();
                head_indices.sort_by(|&a, &b| {
                    weights[start + b]
                        .partial_cmp(&weights[start + a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                head_indices
            })
            .collect();
            
        Ok((weights, indices))
    }
    
    fn route_image(
        &self,
        image_queries: Vec<f32>,
        image_size: (usize, usize),
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let (height, width) = image_size;
        let queries = Array3::from_shape_vec((height, width, self.routing_dim), image_queries)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            
        let num_patches = ((height + 15) / 16) * ((width + 15) / 16);
        
        let results: Vec<_> = (0..num_patches)
            .into_par_iter()
            .map(|p| {
                let patch_y = (p / ((width + 15) / 16)) * 16;
                let patch_x = (p % ((width + 15) / 16)) * 16;
                
                let mut patch_scores = vec![0.0; self.num_heads];
                for h in 0..self.num_heads {
                    let mut score = 0.0;
                    let mut count = 0;
                    
                    for y in patch_y..std::cmp::min(patch_y + 16, height) {
                        for x in patch_x..std::cmp::min(patch_x + 16, width) {
                            score += queries.slice(ndarray::s![y, x, ..]).iter().sum::<f32>();
                            count += self.routing_dim;
                        }
                    }
                    
                    patch_scores[h] = score / (count as f32);
                }
                
                let mut head_indices: Vec<usize> = (0..self.num_heads).collect();
                head_indices.sort_by(|&a, &b| {
                    patch_scores[b]
                        .partial_cmp(&patch_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                
                (patch_scores, head_indices)
            })
            .collect();
            
        let mut weights = Vec::with_capacity(num_patches * self.num_heads);
        let mut indices = Vec::with_capacity(num_patches * self.num_heads);
        
        for (patch_scores, head_indices) in results {
            weights.extend_from_slice(&patch_scores);
            indices.extend_from_slice(&head_indices);
        }
        
        Ok((weights, indices))
    }
    
    fn update_embeddings(
        &self,
        _text_emb: Option<Vec<f32>>,
        _image_emb: Option<Vec<f32>>,
        _text_shape: Option<Vec<usize>>,
        _image_shape: Option<Vec<usize>>,
    ) -> PyResult<()> {
        // Store embeddings for future use if needed
        Ok(())
    }
}
