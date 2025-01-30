use criterion::{black_box, criterion_group, criterion_main, Criterion};
use odysee::{MultiModalProcessor, QuantumState};

fn process_text_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Text Processing");
    let processor = MultiModalProcessor::default();

    group.bench_function("process_text", |b| {
        b.iter(|| {
            let text = black_box("Sample text for benchmarking multimodal processing");
            processor.process_text(text);
        });
    });

    group.finish();
}

fn process_image_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Image Processing");
    let processor = MultiModalProcessor::default();

    // Create a simple test image
    let width = 224;
    let height = 224;
    let channels = 3;
    let image_data = vec![0u8; width * height * channels];

    group.bench_function("process_image", |b| {
        b.iter(|| {
            processor.process_image(black_box(&image_data), width, height);
        });
    });

    group.finish();
}

fn quantum_state_conversion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantum State Conversion");
    let processor = MultiModalProcessor::default();

    // Create sample data
    let text = "Sample text for quantum state conversion";
    let text_state = processor.process_text(text);

    group.bench_function("state_conversion", |b| {
        b.iter(|| {
            QuantumState::from_multimodal(black_box(&text_state));
        });
    });

    group.finish();
}

fn cross_modal_fusion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cross-Modal Fusion");
    let processor = MultiModalProcessor::default();

    // Create sample states
    let text_state = processor.process_text("Text description");
    let width = 224;
    let height = 224;
    let channels = 3;
    let image_data = vec![0u8; width * height * channels];
    let image_state = processor.process_image(&image_data, width, height);

    group.bench_function("fusion", |b| {
        b.iter(|| {
            processor.fuse_modalities(black_box(&text_state), black_box(&image_state));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    process_text_benchmark,
    process_image_benchmark,
    quantum_state_conversion_benchmark,
    cross_modal_fusion_benchmark
);
criterion_main!(benches);
