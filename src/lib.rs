mod quantum_router;
pub use quantum_router::*;

use pyo3::prelude::*;

#[pymodule]
fn odysee(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<quantum_router::QuantumState>()?;
    m.add_class::<quantum_router::QuantumGate>()?;
    m.add_class::<quantum_router::QuantumInspiredRouter>()?;
    m.add_class::<quantum_router::DensityMatrix>()?;
    Ok(())
}
