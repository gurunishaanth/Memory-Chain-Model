import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
    fashion = datasets.FashionMNIST(root='../data', train=train, download=True, transform=transform)
    if max_samples is not None:
        fashion = torch.utils.data.Subset(fashion, list(range(min(max_samples, len(fashion)))))

    data = torch.stack([image.view(-1) for image, _ in fashion], dim=0)
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


def plot_memory_chain(model, min_weight=0.01, out_path='memory_chain.png'):
    if model.num_patterns == 0:
        print("No patterns available for chain plot.")
        return

    transitions = model.T[:model.num_patterns, :model.num_patterns].detach().cpu()
    edges = []
    for i in range(model.num_patterns):
        row = transitions[i]
        if row.sum() <= 0:
            continue
        top_j = torch.argmax(row).item()
        weight = row[top_j].item()
        if weight >= min_weight:
            edges.append((i, top_j, weight))

    if not edges and model.num_patterns > 1:
        for i in range(model.num_patterns - 1):
            edges.append((i, i + 1, 1.0))

    positions = [(i, 0) for i in range(model.num_patterns)]

    num_cols = math.ceil(math.sqrt(model.num_patterns))
    num_rows = math.ceil(model.num_patterns / num_cols)
    canvas = torch.zeros((num_rows * 28, num_cols * 28), dtype=torch.float32)
    patterns = model.W[:model.num_patterns].detach().cpu()
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
        ax_graph.arrow(x0, y0, x1 - x0, y1 - y0, length_includes_head=True, head_width=0.05, color='gray')
        ax_graph.text((x0 + x1) / 2, 0.05, f"{score:.2f}", ha='center', va='bottom', fontsize=8)

    for idx, (x, y) in enumerate(positions):
        ax_graph.scatter([x], [y], s=200, color='skyblue', edgecolors='black', zorder=3)
        ax_graph.text(x, y - 0.1, str(idx), ha='center', va='top', fontsize=9, zorder=4)

    ax_graph.set_title('Memory chain transitions')
    ax_graph.set_xticks(range(model.num_patterns))
    ax_graph.set_yticks([])
    ax_graph.set_xlim(-1, model.num_patterns)
    ax_graph.set_ylim(-1, 1)

    ax_images = fig.add_subplot(1, 2, 2)
    ax_images.imshow(canvas, cmap='gray')
    ax_images.set_title('All learned pattern images')
    ax_images.axis('off')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved memory chain to {out_path}")


def save_pattern_gif(model, out_path='learned_patterns.gif', duration=200):
    if model.num_patterns == 0:
        print("No patterns available for GIF.")
        return

    patterns = model.W[:model.num_patterns].detach().cpu()
    frames = []
    for idx in range(model.num_patterns):
        image = patterns[idx].view(28, 28)
        image = (image - image.min()) / (image.max() - image.min() + 1e-9)
        pixels = (image.numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(pixels, mode='L'))

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"Saved pattern GIF to {out_path}")


def save_generated_sequence(model, start_pattern=0, length=5, prev_count=5, randomize=False, temperature=1.0, out_path='mnist_generated.png'):
    if model.num_patterns == 0:
        print("No patterns available for generation.")
        return

    save_previous_patterns(model, count=prev_count)

    start_pattern = min(max(int(start_pattern), 0), model.num_patterns - 1)
    sequence = model.generate_dream_sequence(start_pattern, length, randomize=randomize, temperature=temperature)
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


def run_memory_train(dataset, input_size, pattern_size, learning_rate, sim_thr, ano_thr, shuffle_dataset=False):
    if shuffle_dataset:
        perm = torch.randperm(dataset.size(0), device=dataset.device)
        dataset = dataset[perm]

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


def save_subchains(model, dataset, out_path='subchains.pt'):
    groups = model.group_dataset_by_pattern(dataset)
    serializable = {
        str(pattern_id): {
            'indices': indices.detach().cpu(),
            'samples': group.detach().cpu(),
        }
        for pattern_id, (indices, group) in groups.items()
    }
    torch.save(serializable, out_path)
    print(f"Saved grouped subchains to {out_path}")
    for pattern_id, (indices, group) in groups.items():
        print(f"pattern {pattern_id}: {group.size(0)} samples")
    return groups


def plot_subchains_on_chain(model, dataset, out_path='memory_chain_with_subchains.png'):
    if model.num_patterns == 0:
        print("No patterns available for subchain plot.")
        return

    groups = model.group_dataset_by_pattern(dataset)
    if not groups:
        print("No grouped subchains found.")
        return

    n = dataset.size(0)
    pattern_ids = sorted(groups.keys())
    offsets = {}
    for i, pattern_id in enumerate(pattern_ids):
        level = (i // 2) + 1
        offsets[pattern_id] = float(level) if i % 2 == 0 else float(-level)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot([0, n - 1], [0, 0], color='black', linewidth=1, alpha=0.6, label='main chain')

    for pattern_id, (indices, _) in groups.items():
        x_vals = indices.cpu().numpy()
        y_val = offsets[pattern_id]
        ax.scatter(x_vals, [y_val] * len(x_vals), label=f'pattern {pattern_id}', s=40)
        if len(x_vals) > 1:
            sorted_x = sorted(x_vals.tolist())
            ax.plot(sorted_x, [y_val] * len(sorted_x), color='gray', alpha=0.5)

    ax.set_xlabel('sample index')
    ax.set_ylabel('subchain offset')
    ax.set_title('Memory chain with up/down subchains grouped by pattern id')
    ax.set_yticks(sorted(set(offsets.values())))
    ax.set_ylim(min(offsets.values()) - 1, max(offsets.values()) + 1)
    ax.set_xlim(-1, n)
    ax.legend(loc='upper right', bbox_to_anchor=(1.14, 1.02), fontsize='small')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved memory chain with subchains plot to {out_path}")


def plot_memory_chain_with_circles(model, dataset, out_path='memory_chain_bigcircles.png', min_weight=0.01):
    if model.num_patterns == 0:
        print("No patterns available for big-circle chain plot.")
        return

    groups = model.group_dataset_by_pattern(dataset)
    if not groups:
        print("No grouped subchains found.")
        return

    transitions = model.T[:model.num_patterns, :model.num_patterns].detach().cpu()
    pattern_ids = sorted(groups.keys())
    positions = {pattern_id: (i * 4.0, 0.0) for i, pattern_id in enumerate(pattern_ids)}

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot([positions[pid][0] for pid in pattern_ids], [0] * len(pattern_ids), color='black', linewidth=1, alpha=0.5)

    for pattern_id in pattern_ids:
        row = transitions[pattern_id]
        if row.sum() <= 0:
            continue
        top_j = torch.argmax(row).item()
        weight = row[top_j].item()
        if weight >= min_weight and top_j in positions:
            x0, y0 = positions[pattern_id]
            x1, y1 = positions[top_j]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='gray', lw=1 + weight * 0.5, alpha=0.7))

    for pattern_id, (indices, _) in groups.items():
        x, y = positions[pattern_id]
        count = len(indices)
        circle_size = 600 + count * 40
        ax.scatter([x], [y], s=circle_size, color='skyblue', edgecolors='black', linewidths=1.5, alpha=0.8, zorder=3)
        ax.text(x, y, f'{pattern_id}\n{count}', ha='center', va='center', fontsize=10, weight='bold')

        if count > 0:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            radius = 1.2
            xs = x + radius * np.cos(angles)
            ys = y + radius * np.sin(angles)
            ax.scatter(xs, ys, color='red', s=20, alpha=0.7, zorder=4)
            for xi, yi in zip(xs, ys):
                ax.plot([x, xi], [y, yi], color='red', alpha=0.15, linewidth=0.8)

    ax.set_title('Memory chain transitions with grouped data')
    ax.set_xlabel('pattern position')
    ax.set_yticks([])
    ax.set_xlim(-2, positions[pattern_ids[-1]][0] + 2)
    ax.set_ylim(-4, 4)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved big-circle memory chain plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='MemoryChain MNIST training example')
    parser.add_argument('--mnist', action='store_true', help='Train on FashionMNIST instead of the toy sparse dataset')
    parser.add_argument('--samples', type=int, default=2000, help='Number of FashionMNIST samples to use')
    parser.add_argument('--patterns', type=int, default=500, help='Maximum number of memory patterns')
    parser.add_argument('--sim-thr', type=float, default=50.0, help='Similarity threshold for new pattern creation')
    parser.add_argument('--ano-thr', type=float, default=20.0, help='Anomaly threshold')
    parser.add_argument('--start-pattern', type=int, default=0, help='Starting pattern index for generation')
    parser.add_argument('--generate-length', type=int, default=5, help='Number of next patterns to generate')
    parser.add_argument('--show-prev', type=int, default=5, help='Number of previous patterns to display before generation')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle input order to form a chain rather than a fixed sequence')
    parser.add_argument('--randomize-generation', action='store_true', help='Randomize generation from chain transitions')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for randomized generation')
    parser.add_argument('--save-gif', action='store_true', help='Save a GIF of all learned patterns')
    parser.add_argument('--save-chain', type=str, default='', help='Save the learned temporal chain data to a file')
    parser.add_argument('--save-subchains', type=str, default='', help='Save grouped subchains by pattern id to a file')
    parser.add_argument('--plot-subchains', action='store_true', help='Plot grouped subchains on the memory chain')
    parser.add_argument('--plot-bigcircles', action='store_true', help='Plot memory chain transitions with big pattern circles and grouped data')
    parser.add_argument('--tree-thr', type=float, default=0.3, help='Minimum transition weight to show in chain plot')
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
            shuffle_dataset=args.shuffle,
        )
        plot_memory_results(results)
        plot_memory_chain(model, min_weight=args.tree_thr)
        if args.save_gif:
            save_pattern_gif(model, out_path='learned_patterns.gif')
        if args.save_chain:
            saved_path = model.save_temporal_chain(args.save_chain)
            print(f"Saved temporal chain data to {saved_path}")
        if args.save_subchains:
            save_subchains(model, dataset, out_path=args.save_subchains)
        if args.plot_subchains:
            plot_subchains_on_chain(model, dataset, out_path='memory_chain_with_subchains.png')
        if args.plot_bigcircles:
            plot_memory_chain_with_circles(model, dataset, out_path='memory_chain_bigcircles.png')
        save_generated_sequence(
            model,
            start_pattern=args.start_pattern,
            length=args.generate_length,
            prev_count=args.show_prev,
            randomize=args.randomize_generation,
            temperature=args.temperature,
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
