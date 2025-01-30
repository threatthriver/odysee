use criterion::{black_box, criterion_group, criterion_main, Criterion};
use odysee::{DistributedMemory, MultiModalProcessor};

fn memory_store_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Store");
    let processor = MultiModalProcessor::default();
    let memory = DistributedMemory::new(1_000_000);

    group.bench_function("store_text", |b| {
        b.iter(|| {
            let text = black_box("Sample text for benchmarking");
            let state = processor.process_text(text);
            memory.store_multimodal(1, state);
        });
    });

    group.finish();
}

fn memory_retrieve_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Retrieve");
    let processor = MultiModalProcessor::default();
    let memory = DistributedMemory::new(1_000_000);

    // Setup: Store some data first
    let text = "Sample text for benchmarking";
    let state = processor.process_text(text);
    memory.store_multimodal(1, state);

    group.bench_function("retrieve_text", |b| {
        b.iter(|| {
            memory.retrieve_multimodal(black_box(1));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    memory_store_benchmark,
    memory_retrieve_benchmark
);
criterion_main!(benches);
