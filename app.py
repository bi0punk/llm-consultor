#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

# -----------------------------
# HW detection
# -----------------------------
def read_meminfo_kb() -> Dict[str, int]:
    out = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^(\w+):\s+(\d+)\s+kB", line.strip())
            if m:
                out[m.group(1)] = int(m.group(2))
    return out

def gb(x_kb: int) -> float:
    return x_kb / (1024 * 1024)

def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        return m.group(1).strip() if m else "unknown"
    except Exception:
        return "unknown"

def cpu_flags() -> List[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.search(r"flags\s*:\s*(.+)", txt)
        if not m:
            return []
        return m.group(1).split()
    except Exception:
        return []

def cores_threads() -> Dict[str, int]:
    # Linux: os.cpu_count() = threads lógicos
    threads = os.cpu_count() or 1
    # cores físicos no siempre fácil sin psutil; aproximamos
    return {"threads_logical": threads}

def nvidia_vram_gb() -> Optional[float]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        # total VRAM in MiB
        cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        if not out:
            return None
        # si hay varias GPUs, tomamos la mayor
        mib = max(int(x.strip()) for x in out if x.strip().isdigit())
        return mib / 1024.0
    except Exception:
        return None

# -----------------------------
# Estimación de memoria
# -----------------------------
# Memoria modelo por 1B params (GB) según quant (aprox GGUF)
Q_GB_PER_1B = {
    "Q4": 0.55,
    "Q5": 0.68,
    "Q8": 1.00,
}

# KV cache por token (KB/token) para familias Llama-like (aprox, fp16)
# Esto es “consultoría” práctica: no es exacto por arquitectura, pero sirve para decidir.
KV_KB_PER_TOKEN = {
    "8B": 128,
    "13B": 200,
    "34B": 520,
    "70B": 1024,
}

FAMILIES = [
    ("8B", 8),
    ("13B", 13),
    ("34B", 34),
    ("70B", 70),
]

@dataclass
class Recommendation:
    family: str
    params_b: int
    quant: str
    ctx: int
    est_model_gb: float
    est_kv_gb: float
    est_total_gb: float
    ram_available_gb: float
    margin_gb: float
    semaforo: str
    reason: str
    gpu_fit: Optional[str]  # texto simple si hay VRAM

def semaforo_from_margin(margin_gb: float) -> str:
    if margin_gb >= 2.5:
        return "🟢"
    if margin_gb >= 1.0:
        return "🟡"
    return "🔴"

def estimate_model_gb(params_b: int, quant: str) -> float:
    base = params_b * Q_GB_PER_1B[quant]
    return base * 1.15  # +15% overhead/buffers/mmap

def estimate_kv_gb(family: str, ctx: int) -> float:
    kb_per_tok = KV_KB_PER_TOKEN[family]
    total_kb = kb_per_tok * ctx
    return total_kb / (1024 * 1024)  # KB -> GB

def gpu_fit_text(vram_gb: float, total_gb: float) -> str:
    # simplificado: si VRAM >= total_gb => full offload potencial (depende runtime)
    if vram_gb >= total_gb + 1.0:
        return f"VRAM OK (≈{vram_gb:.1f} GB) para offload casi total"
    if vram_gb >= (0.60 * total_gb):
        return f"VRAM parcial (≈{vram_gb:.1f} GB): offload parcial, mejora tok/s"
    return f"VRAM baja (≈{vram_gb:.1f} GB): probablemente CPU mostly"

def main():
    ap = argparse.ArgumentParser(description="Consultor LLM: recomienda tamaños/quant sin descargar modelos.")
    ap.add_argument("--ctx", type=int, default=4096, help="Contexto objetivo (tokens). Ej: 2048/4096/8192")
    ap.add_argument("--only", default="", help="Filtra familias: 8B,13B,34B,70B (coma-separado)")
    ap.add_argument("--json-out", default="", help="Si se indica, guarda resultado JSON")
    args = ap.parse_args()

    mi = read_meminfo_kb()
    ram_total = gb(mi.get("MemTotal", 0))
    ram_avail = gb(mi.get("MemAvailable", 0))

    flags = set(cpu_flags())
    vram = nvidia_vram_gb()

    sysinfo = {
        "cpu": cpu_model(),
        "threads_logical": cores_threads()["threads_logical"],
        "ram_total_gb": round(ram_total, 2),
        "ram_available_gb": round(ram_avail, 2),
        "cpu_flags_has_avx2": ("avx2" in flags),
        "cpu_flags_has_fma": ("fma" in flags),
        "nvidia_vram_gb": None if vram is None else round(vram, 2),
        "ctx_target": args.ctx,
    }

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    families = [f for f in FAMILIES if (not only or f[0] in only)]

    recs: List[Recommendation] = []
    for fam, pb in families:
        for quant in ("Q4", "Q5", "Q8"):
            est_model = estimate_model_gb(pb, quant)
            est_kv = estimate_kv_gb(fam, args.ctx)
            # margen anti-swap extra (SO + caches + margen humano)
            safety = 1.5
            est_total = est_model + est_kv + safety

            margin = ram_avail - est_total
            sem = semaforo_from_margin(margin)

            reason = f"OK para probar" if sem != "🔴" else "Riesgo de swap/OOM o inutilmente lento"
            if sem == "🟡":
                reason = "Corre, pero justo: baja ctx o usa Q4"
            if fam in ("34B", "70B") and quant in ("Q8", "Q5") and sem != "🟢":
                reason += " (estos tamaños suelen exigir mucha RAM)"

            gpu_txt = None
            if vram is not None:
                gpu_txt = gpu_fit_text(vram, est_total)

            recs.append(Recommendation(
                family=fam,
                params_b=pb,
                quant=quant,
                ctx=args.ctx,
                est_model_gb=round(est_model, 2),
                est_kv_gb=round(est_kv, 2),
                est_total_gb=round(est_total, 2),
                ram_available_gb=round(ram_avail, 2),
                margin_gb=round(margin, 2),
                semaforo=sem,
                reason=reason,
                gpu_fit=gpu_txt,
            ))

    # Orden: primero 🟢, luego 🟡, luego 🔴, y por mayor margen
    order = {"🟢": 0, "🟡": 1, "🔴": 2}
    recs_sorted = sorted(recs, key=lambda r: (order[r.semaforo], -r.margin_gb))

    print("=== HW ===")
    print(json.dumps(sysinfo, indent=2, ensure_ascii=False))

    print("\n=== Recomendación (sin descargar modelos) ===")
    # imprime top por familia (mejor quant)
    best_by_family: Dict[str, Recommendation] = {}
    for r in recs_sorted:
        if r.family not in best_by_family:
            best_by_family[r.family] = r

    for fam in [x[0] for x in families]:
        r = best_by_family.get(fam)
        if not r:
            continue
        print(f"\n[{fam}] mejor apuesta: {r.semaforo} {r.quant} @ ctx={r.ctx}")
        print(f"  Est modelo: {r.est_model_gb} GB | KV: {r.est_kv_gb} GB | Total+seg: {r.est_total_gb} GB")
        print(f"  RAM disponible: {r.ram_available_gb} GB | Margen: {r.margin_gb} GB -> {r.reason}")
        if r.gpu_fit:
            print(f"  GPU: {r.gpu_fit}")

    # tabla completa resumida
    print("\n=== Tabla completa (ordenada) ===")
    for r in recs_sorted:
        print(f"{r.semaforo} {r.family:>3} {r.quant} | total≈{r.est_total_gb:>6}GB | margen={r.margin_gb:>6}GB")

    if args.json_out:
        payload = {"system": sysinfo, "recommendations": [asdict(r) for r in recs_sorted]}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nOK -> Guardado JSON: {args.json_out}")

if __name__ == "__main__":
    main()
