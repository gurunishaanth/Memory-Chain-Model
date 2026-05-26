import argparse
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import Memory_cell


def build_sparse_dataset():
    dataset_dicts = [
        {0:1, 1:1, 2:1},
        {0:1, 1:1, 3:1},
        {0:1, 1:1, 2:1},
        {5:1, 6:1},
        {5:1, 6:1, 7:1},
        {0:1, 1:1, 2:1},
        {20:1},          # anomaly-like
        {5:1, 6:1},
    ]

    dataset = torch.zeros((len(dataset_dicts), 21), dtype=torch.float32)
    for i, item in enumerate(dataset_dicts):
        for idx, value in item.items():
            dataset[i, int(idx)] = value
    return dataset


def load_mnist_dataset(max_samples=None, train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='../data', train=train, download=True, transform=transform)
    if max_samples is not None:
        mnist = torch.utils.data.Subset(mnist, list(range(min(max_samples, len(mnist)))))

    data = torch.stack([image.view(-1) for image, _ in mnist], dim=0)
    return data


def plot_memory_results(results, out_path='memory_chain_plot.png'):
    t = [r[0] for r in results]
    k = [r[1] for r in results]
    conf = [r[2] for r in results]
    anomaly = [r[3] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t, k, label='pattern index k', marker='o')
    ax1.set_xlabel('time step t')
    ax1.set_ylabel('pattern index k')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t, conf, color='tab:green', label='confidence', linestyle='--')
    ax2.set_ylabel('confidence')

    anomaly_t = [r[0] for r in results if r[3]]
    anomaly_conf = [r[2] for r in results if r[3]]
    if anomaly_t:
        ax2.scatter(anomaly_t, anomaly_conf, color='red', marker='x', label='anomaly', zorder=5)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def save_previous_patterns(model, count=5, out_path='previous_patterns.png'):
    if model.num_patterns == 0:
        print("No previous patterns to display.")
        return

    end = model.num_patterns
    start = max(0, end - count)
    indices = list(range(start, end))
    print(f"Displaying previous {len(indices)} patterns: {indices}")

    images = [model.recall_pattern(idx).detach().cpu().numpy().reshape(28, 28) for idx in indices]
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    if len(images) == 1:
        axes = [axes]

    for ax, img, idx in zip(axes, images, indices):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'pattern {idx}')
        ax.axis('off')

    fig.suptitle(f'Previous {len(indices)} learned MNIST patterns')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved previous patterns to {out_path}")


def plot_memory_tree(model, similarity_threshold=0.3, out_path='memory_tree.png'):
    if model.num_patterns == 0:
        print("No patterns available for tree plot.")
        return

    patterns = model.W[:model.num_patterns].detach().cpu()
    norms = torch.norm(patterns, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    sim = (patterns @ patterns.T) / (norms @ norms.T)

    edges = []
    for i in range(model.num_patterns):
        for j in range(i + 1, model.num_patterns):
            if sim[i, j].item() >= similarity_threshold:
                edges.append((i, j, sim[i, j].item()))

    if not edges and model.num_patterns > 1:
        for i in range(model.num_patterns):
            row = sim[i].clone()
            row[i] = -1.0
            j = torch.argmax(row).item()
            edges.append((i, j, sim[i, j].item()))

    theta = [2 * math.pi * i / model.num_patterns for i in range(model.num_patterns)]
    positions = [(math.cos(t), math.sin(t)) for t in theta]

    num_cols = math.ceil(math.sqrt(model.num_patterns))
    num_rows = math.ceil(model.num_patterns / num_cols)
    canvas = torch.zeros((num_rows * 28, num_cols * 28), dtype=torch.float32)
    for idx in range(model.num_patterns):
        row = idx // num_cols
        col = idx % num_cols
        image = patterns[idx].view(28, 28)
        image = (image - image.min()) / (image.max() - image.min() + 1e-9)
        canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = image

    fig = plt.figure(figsize=(16, 8))
    ax_graph = fig.add_subplot(1, 2, 1)
    for i, j, score in edges:
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        ax_graph.plot([x0, x1], [y0, y1], color='gray', linewidth=1)

    for idx, (x, y) in enumerate(positions):
        ax_graph.scatter([x], [y], s=200, color='skyblue', edgecolors='black', zorder=3)
        ax_graph.text(x, y, str(idx), ha='center', va='center', fontsize=9, zorder=4)

    ax_graph.set_title('Memory similarity tree')
    ax_graph.set_xticks([])
    ax_graph.set_yticks([])
    ax_graph.set_aspect('equal')

    ax_images = fig.add_subplot(1, 2, 2)
    ax_images.imshow(canvas, cmap='gray')
    ax_images.set_title('All learned pattern images')
    ax_images.axis('off')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved memory tree to {out_path}")


def save_generated_sequence(model, start_pattern=0, length=5, prev_count=5, out_path='mnist_generated.png'):
    if model.num_patterns == 0:
        print("No patterns available for generation.")
        return

    save_previous_patterns(model, count=prev_count)

    start_pattern = min(max(int(start_pattern), 0), model.num_patterns - 1)
    sequence = model.generate_dream_sequence(start_pattern, length)
    print(f"Predicted next {length} pattern sequence from pattern {start_pattern}: {sequence}")

    images = [model.generate(idx).detach().cpu().numpy().reshape(28, 28) for idx in sequence]

    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    if len(images) == 1:
        axes = [axes]

    for ax, img, idx in zip(axes, images, sequence):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'pattern {idx}')
        ax.axis('off')

    fig.suptitle(f'Generated MNIST pattern sequence (next {length})')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved generated sequence to {out_path}")


def run_memory_train(dataset, input_size, pattern_size, learning_rate, sim_thr, ano_thr):
    model = Memory_cell.MemoryCell(
        input_size=input_size,
        pattern_size=pattern_size,
        learning_rate=learning_rate,
        Sim_Thr=sim_thr,
        Ano_Thr=ano_thr,
    )

    results = []
    for t, x in enumerate(dataset):
        k, conf, anomaly = model.Mem_chain_step(x)
        results.append((t, k, round(conf, 2), anomaly))

    print(f"learned patterns: {model.num_patterns}")
    print(f"anomaly rate: {sum(1 for _, _, _, anomaly in results if anomaly)/len(results):.2%}")
    print("first 20 results:")
    for entry in results[:20]:
        print(entry)
    return model, results


def main():
    parser = argparse.ArgumentParser(description='MemoryChain MNIST training example')
    parser.add_argument('--mnist', action='store_true', help='Train on MNIST instead of the toy sparse dataset')
    parser.add_argument('--samples', type=int, default=2000, help='Number of MNIST samples to use')
    parser.add_argument('--patterns', type=int, default=500, help='Maximum number of memory patterns')
    parser.add_argument('--sim-thr', type=float, default=50.0, help='Similarity threshold for new pattern creation')
    parser.add_argument('--ano-thr', type=float, default=20.0, help='Anomaly threshold')
    parser.add_argument('--start-pattern', type=int, default=0, help='Starting pattern index for generation')
    parser.add_argument('--generate-length', type=int, default=5, help='Number of next patterns to generate')
    parser.add_argument('--show-prev', type=int, default=5, help='Number of previous patterns to display before generation')
    parser.add_argument('--tree-thr', type=float, default=0.3, help='Similarity threshold for memory tree edges')
    args = parser.parse_args()

    if args.mnist:
        dataset = load_mnist_dataset(max_samples=args.samples, train=True)
        input_size = 28 * 28
        model, results = run_memory_train(
            dataset=dataset,
            input_size=input_size,
            pattern_size=args.patterns,
            learning_rate=0.01,
            sim_thr=args.sim_thr,
            ano_thr=args.ano_thr,
        )
        plot_memory_results(results)
        plot_memory_tree(model, similarity_threshold=args.tree_thr)
        save_generated_sequence(
            model,
            start_pattern=args.start_pattern,
            length=args.generate_length,
            prev_count=args.show_prev,
        )
    else:
        dataset = build_sparse_dataset()
        run_memory_train(
            dataset=dataset,
            input_size=21,
            pattern_size=10,
            learning_rate=0.1,
            sim_thr=0.5,
            ano_thr=0.3,
        )


if __name__ == '__main__':
    main()
