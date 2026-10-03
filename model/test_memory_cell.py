"""Behavioural tests for MemoryCell. Run: python3 test_memory_cell.py"""
import importlib
import inspect
import os
import tempfile

import torch

# Set CELL=<module name> to run the suite against a different implementation
MemoryCell = importlib.import_module(os.environ.get("CELL", "Memory_cell")).MemoryCell

torch.manual_seed(0)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))


def make(dim=16, n=10, lr=0.1, sim=0.5, ano=0.3):
    kw = {"cosine": True} if "cosine" in inspect.signature(MemoryCell.__init__).parameters else {}
    return MemoryCell(input_size=dim, pattern_size=n, learning_rate=lr, Sim_Thr=sim, Ano_Thr=ano, **kw)


def onehot_block(i, dim=16, width=3):
    x = torch.zeros(dim)
    x[i * width:(i + 1) * width] = 1.0
    return x


A, B, C = onehot_block(0), onehot_block(1), onehot_block(2)


def test_chain_direction():
    m = make()
    for _ in range(3):
        for x in (A, B, C):
            m.Mem_chain_step(x)
    kA, kB, kC = [m.forward(x)[1].item() for x in (A, B, C)]
    pred = m.predict_next(kA)
    check("1. predict_next(A) returns B (forward in time)", pred == kB,
          f"A={kA} B={kB} C={kC}; predict_next(A)={pred}; T=\n{m.T[:3, :3]}")
    seq = m.generate_dream_sequence(kA, 4)
    check("2. dream sequence from A follows A->B->C->A", seq == [kA, kB, kC, kA], f"got {seq}")


def test_hebbian_averaging():
    m = make(dim=16, sim=0.6)
    proto = A.clone()
    for _ in range(50):
        m.Mem_chain_step(proto + 0.3 * torch.randn(16))
    k = m.forward(proto)[1].item()
    err_mem = torch.norm(m.W[k] - proto).item()
    noise_level = 0.3 * 16 ** 0.5
    check("3. stored pattern converges toward the noiseless prototype (Hebbian averaging)",
          err_mem < 0.5 * noise_level,
          f"||W_k - prototype|| = {err_mem:.2f}; a single noisy sample is ~{noise_level:.2f} away. "
          f"patterns created: {m.num_patterns}")


def test_scale_bias():
    m = make(sim=0.5)
    bright = A * 5.0
    m.Mem_chain_step(bright)
    m.Mem_chain_step(B)
    # B and a slight B variant should match B's pattern, not the bright A pattern
    probe = B.clone(); probe[0] = 1.0  # B plus one overlapping A feature
    a, s = m.forward(probe)
    kB = 1
    check("4. matching is scale-invariant (probe close to B matches B, not a bright A)",
          s.item() == kB, f"activations={a[0, :2].tolist()} -> picked pattern {s.item()}")


def test_capacity_replacement():
    m = make(dim=40, n=2, sim=0.5)
    X = [onehot_block(i, dim=40, width=4) for i in range(3)]
    for x in X:
        m.Mem_chain_step(x)  # X0->slot0, X1->slot1, X2 overwrites a slot
    # Only valid link among stored items is X1 -> X2; the old X0<->X1 link should be gone
    n_links = round(m.T.sum().item() / m.learning_rate)
    check("5. replacing a pattern when memory is full clears its old transitions",
          n_links == 1, f"expected 1 live link, found {n_links}; T=\n{m.T}")


def test_persistence_across_sessions():
    m = make()
    for _ in range(3):
        for x in (A, B, C):
            m.Mem_chain_step(x)
    path = os.path.join(tempfile.mkdtemp(), "chain.pt")
    m.save_temporal_chain(path)
    try:
        m2 = MemoryCell.load_temporal_chain(path, 16, 10, 0.1, 0.5, 0.3)
    except RuntimeError as e:
        check("6. a reloaded 'new session' recalls the same chain", False, f"load crashed: {e}")
        return
    same_W = torch.allclose(m.W, m2.W)
    same_T = torch.allclose(m.T, m2.T)
    k1 = m.predict_next(m.forward(A)[1].item())
    k2 = m2.predict_next(m2.forward(A)[1].item())
    check("6. a reloaded 'new session' recalls the same chain", same_W and same_T and k1 == k2,
          f"W equal={same_W}, T equal={same_T}, prediction {k1} vs {k2}")
    # does the new session continue the chain or start a disconnected one?
    m2.Mem_chain_step(A)
    check("7. reloaded session starts with no stale prev_k (no false link into new data)",
          m2.T.sum().item() == m.T.sum().item(), "first step after reload added a link")


def test_anomaly():
    m = make()
    for _ in range(3):
        for x in (A, B):
            m.Mem_chain_step(x)
    weird = torch.zeros(16); weird[15] = 1.0
    _, conf, anomaly = m.Mem_chain_step(weird)
    check("8. unseen input flagged as anomaly", anomaly, f"confidence={conf}")
    _, conf2, anomaly2 = m.Mem_chain_step(A)
    check("9. familiar input not flagged", not anomaly2, f"confidence={conf2}")


if __name__ == "__main__":
    for t in (test_chain_direction, test_hebbian_averaging, test_scale_bias,
              test_capacity_replacement, test_persistence_across_sessions, test_anomaly):
        t()
    passed = sum(r[1] for r in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} passed")
