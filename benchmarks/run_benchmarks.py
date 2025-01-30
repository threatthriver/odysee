import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from odysee import QuantumInspiredRouter, HierarchicalMemory

def benchmark_routing(batch_sizes, seq_lengths, dim=1024, num_heads=8):
    results = {}
    router = QuantumInspiredRouter(dim=dim, num_heads=num_heads)
    
    for batch_size in tqdm(batch_sizes, desc="Batch sizes"):
        results[batch_size] = {}
        for seq_len in tqdm(seq_lengths, desc="Sequence lengths"):
            # Generate random input
            x = np.random.randn(batch_size, seq_len, dim)
            
            # Warmup
            for _ in range(3):
                router.forward(x[0])
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.perf_counter()
                router.forward(x)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            throughput = (batch_size * seq_len) / avg_time
            results[batch_size][seq_len] = {
                'time': avg_time,
                'throughput': throughput
            }
    
    return results

def benchmark_memory(sizes, dim=1024):
    results = {}
    memory = HierarchicalMemory(capacity=max(sizes) * dim)
    
    for size in tqdm(sizes, desc="Memory sizes"):
        # Generate random data
        data = np.random.randn(size, dim)
        importance = np.random.rand(size)
        
        # Benchmark storage
        start = time.perf_counter()
        for i in range(size):
            memory.store(data[i:i+1], importance[i])
        store_time = time.perf_counter() - start
        
        # Generate queries
        queries = np.random.randn(100, dim)
        
        # Benchmark retrieval
        times = []
        for query in queries:
            start = time.perf_counter()
            memory.retrieve(query.reshape(1, -1))
            times.append(time.perf_counter() - start)
        
        retrieval_time = np.mean(times)
        
        results[size] = {
            'store_time': store_time,
            'retrieval_time': retrieval_time
        }
    
    return results

def plot_results(routing_results, memory_results):
    # Routing plots
    plt.figure(figsize=(15, 5))
    
    # Throughput plot
    plt.subplot(121)
    for batch_size, results in routing_results.items():
        seq_lengths = list(results.keys())
        throughputs = [r['throughput'] for r in results.values()]
        plt.plot(seq_lengths, throughputs, label=f'Batch {batch_size}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Routing Throughput')
    plt.legend()
    plt.grid()
    
    # Memory plots
    plt.subplot(122)
    sizes = list(memory_results.keys())
    store_times = [r['store_time'] for r in memory_results.values()]
    retrieval_times = [r['retrieval_time'] for r in memory_results.values()]
    
    plt.plot(sizes, store_times, label='Store')
    plt.plot(sizes, retrieval_times, label='Retrieve')
    plt.xlabel('Memory Size (tokens)')
    plt.ylabel('Time (s)')
    plt.title('Memory Performance')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('benchmarks.png')
    plt.close()

def main():
    print("Running Odysee Benchmarks")
    print("========================")
    
    # Routing benchmarks
    print("\nRouting Benchmarks:")
    batch_sizes = [1, 8, 32, 128]
    seq_lengths = [128, 512, 2048, 8192]
    routing_results = benchmark_routing(batch_sizes, seq_lengths)
    
    # Memory benchmarks
    print("\nMemory Benchmarks:")
    sizes = [1000, 10000, 100000, 1000000]
    memory_results = benchmark_memory(sizes)
    
    # Plot results
    plot_results(routing_results, memory_results)
    
    # Print summary
    print("\nSummary:")
    print("--------")
    
    # Routing summary
    max_throughput = 0
    for batch_results in routing_results.values():
        for result in batch_results.values():
            max_throughput = max(max_throughput, result['throughput'])
    
    print(f"Maximum Routing Throughput: {max_throughput:.2f} tokens/s")
    
    # Memory summary
    max_size = max(sizes)
    final_retrieval_time = memory_results[max_size]['retrieval_time']
    print(f"Memory Retrieval Time ({max_size} tokens): {final_retrieval_time*1000:.2f} ms")

if __name__ == "__main__":
    main()
