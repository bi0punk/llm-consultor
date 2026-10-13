#!/usr/bin/env python3

"""
llm_machine_advisor.py
Analiza CPU/RAM/SWAP y recomienda modelos GGUF (llama.cpp / Ollama / LocalAI)
con clasificación: PERFECTO / BIEN / AJUSTADO / MAL.

Enfoque:
- Pesos: params * (bits_per_weight/8)
- KV cache: heurística por (layers * hidden * ctx * bytes * 2[K+V]) con fallback conservador
- Overhead: base + % del tamaño de pesos

Fuentes (para contexto general de quants/bpw):
- Explicación bpw y quants legacy/k-quants (incluye ejemplo Q4_1 ~4.0625 bpw)
- Librería de modelos populares (tamaños y familias)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass

# Optional deps
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    from tabulate import tabulate  # type: ignore
except Exception:
    tabulate = None


# -----------------------------
# Quantization heuristics (bpw)
# -----------------------------
# bpw: bits per weight (promedio). Para K-quants, valores típicos rondan ~4.6 en Q4KM en algunos modelos (varía).
# Usamos valores conservadores y simples.
BPW_MAP: dict[str, float] = {
    # Legacy
    "Q2_K": 2.6,
    "Q3_K_S": 3.4,
    "Q3_K_M": 3.7,
    "Q3_K_L": 3.9,
    "Q4_0": 4.0,
    "Q4_1": 4.0625,   # citado como ejemplo en explicación de legacy quants
    "Q4_K_S": 4.4,
    "Q4_K_M": 4.65,   # conservador (en la práctica suele estar ~4.6 bpw en varios casos)
    "Q5_0": 5.0,
    "Q5_1": 5.125,
    "Q5_K_S": 5.4,
    "Q5_K_M": 5.6,
    "Q6_K": 6.6,
    "Q8_0": 8.5,      # incluye overhead de escalas/bloques
    "F16": 16.0,
}

DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


# -----------------------------------
# Model catalog (popular local families)
# -----------------------------------
@dataclass(frozen=True)
class ModelSpec:
    family: str
    name: str
    params_b: float                 # billions
    task: str                       # general / coding / reasoning / multilingual / vision
    # Architecture hints for KV cache estimation (best-effort, conservative)
    n_layers: int | None = None
    hidden: int | None = None
    notes: str = ""


# Curado práctico (sizes típicos que ves en catálogos locales)
MODEL_CATALOG: list[ModelSpec] = [
    # Llama
    ModelSpec(
        "meta", "Llama 3.2 Instruct", 1.0, "multilingual",
        n_layers=16, hidden=2048, notes="Muy liviano; ideal CPU",
    ),
    ModelSpec(
        "meta", "Llama 3.2 Instruct", 3.0, "multilingual",
        n_layers=28, hidden=3072, notes="Buen balance en 16GB",
    ),
    ModelSpec("meta", "Llama 3.1 Instruct", 8.0, "general", n_layers=32, hidden=4096, notes="El ‘sweet spot’ típico"),
    ModelSpec("meta", "Llama 3.1 Instruct", 70.0, "general", n_layers=80, hidden=8192, notes="Requiere mucha RAM/VRAM"),

    # Qwen
    ModelSpec(
        "alibaba", "Qwen 2.5 Instruct", 0.5, "multilingual",
        n_layers=24, hidden=1024, notes="Excelente ultra-ligero",
    ),
    ModelSpec("alibaba", "Qwen 2.5 Instruct", 1.5, "multilingual", n_layers=28, hidden=1536, notes="Muy buen básico"),
    ModelSpec("alibaba", "Qwen 2.5 Instruct", 3.0, "multilingual", n_layers=36, hidden=2048, notes="Razonable en CPU"),
    ModelSpec("alibaba", "Qwen 2.5 Instruct", 7.0, "multilingual", n_layers=32, hidden=4096, notes="Fuerte general"),
    ModelSpec("alibaba", "Qwen 2.5 Instruct", 14.0, "multilingual", n_layers=40, hidden=5120, notes="Ya pesado en CPU"),
    ModelSpec(
        "alibaba", "Qwen 2.5 Instruct", 32.0, "multilingual",
        n_layers=64, hidden=5120, notes="Normalmente servidor",
    ),
    ModelSpec("alibaba", "Qwen 2.5 Coder", 7.0, "coding", n_layers=32, hidden=4096, notes="Código en local"),
    ModelSpec("alibaba", "Qwen 2.5 Coder", 14.0, "coding", n_layers=40, hidden=5120, notes="Mejor pero pesado"),
    ModelSpec("alibaba", "Qwen 3", 8.0, "reasoning", n_layers=32, hidden=4096, notes="Más foco en razonamiento"),
    ModelSpec("alibaba", "Qwen 3", 14.0, "reasoning", n_layers=40, hidden=5120, notes="Razonamiento, más RAM"),

    # Mistral
    ModelSpec(
        "mistral", "Mistral 7B Instruct", 7.0, "general",
        n_layers=32, hidden=4096, notes="Rinde muy bien en CPU",
    ),
    ModelSpec("mistral", "Mistral Nemo", 12.0, "general", n_layers=40, hidden=5120, notes="12B con contexto grande"),
    ModelSpec(
        "mistral", "Mixtral 8x7B (MoE)", 46.7, "general",
        n_layers=32, hidden=4096, notes="MoE: pesado en RAM; mejor GPU",
    ),

    # Gemma
    ModelSpec("google", "Gemma 2", 2.0, "general", n_layers=18, hidden=2048, notes="Pequeño y decente"),
    ModelSpec("google", "Gemma 2", 9.0, "general", n_layers=42, hidden=3584, notes="Muy buen general"),
    ModelSpec("google", "Gemma 2", 27.0, "general", n_layers=46, hidden=4608, notes="Servidor/VRAM"),
    ModelSpec("google", "Gemma 3", 4.0, "general", n_layers=28, hidden=3072, notes="Buen balance moderno"),
    ModelSpec("google", "Gemma 3", 12.0, "general", n_layers=42, hidden=4096, notes="Requiere más RAM"),

    # Phi
    ModelSpec("microsoft", "Phi-3 Mini", 3.8, "general", n_layers=32, hidden=3072, notes="Calidad alta por tamaño"),
    ModelSpec("microsoft", "Phi-3 Medium", 14.0, "general", n_layers=40, hidden=5120, notes="Buen salto de calidad"),
    ModelSpec("microsoft", "Phi-4", 14.0, "reasoning", n_layers=40, hidden=5120, notes="Mejor razonamiento; pesado"),

    # DeepSeek (razonamiento)
    ModelSpec(
        "deepseek", "DeepSeek R1 Distill", 1.5, "reasoning",
        n_layers=28, hidden=1536, notes="Razonamiento liviano",
    ),
    ModelSpec("deepseek", "DeepSeek R1", 7.0, "reasoning", n_layers=32, hidden=4096, notes="Razonamiento local"),
    ModelSpec("deepseek", "DeepSeek R1", 14.0, "reasoning", n_layers=40, hidden=5120, notes="Mejor, más RAM"),

    # CodeLlama
    ModelSpec("meta", "CodeLlama Instruct", 7.0, "coding", n_layers=32, hidden=4096, notes="Coding clásico"),
    ModelSpec("meta", "CodeLlama Instruct", 13.0, "coding", n_layers=40, hidden=5120, notes="Más capaz, más pesado"),
    ModelSpec("meta", "CodeLlama Instruct", 34.0, "coding", n_layers=48, hidden=8192, notes="Servidor/GPU normalmente"),

    # TinyLlama
    ModelSpec("community", "TinyLlama Chat", 1.1, "general", n_layers=22, hidden=2048, notes="Ultra-ligero"),
]


# -----------------------------
# System probing
# -----------------------------
@dataclass
class MachineInfo:
    os: str
    arch: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    cpu_mhz: float | None
    cpu_flags: list[str]
    ram_total_gb: float
    ram_available_gb: float
    swap_total_gb: float
    swap_free_gb: float


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return ""


def get_cpu_flags_linux() -> list[str]:
    flags = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    _, v = line.split(":", 1)
                    for fl in v.strip().split():
                        flags.add(fl)
                    break
    except Exception:
        pass
    return sorted(flags)


def get_cpu_model_linux() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_cpu_mhz_linux() -> float | None:
    # Best effort
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("cpu mhz"):
                    return float(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def get_machine_info() -> MachineInfo:
    os_name = f"{platform.system()} {platform.release()}"
    arch = platform.machine()

    if psutil:
        physical = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or 0
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        ram_total_gb = vm.total / (1024**3)
        ram_available_gb = vm.available / (1024**3)
        swap_total_gb = sm.total / (1024**3)
        swap_free_gb = (sm.total - sm.used) / (1024**3)
    else:
        # Fallback minimal
        physical = os.cpu_count() or 0
        logical = physical
        ram_total_gb = ram_available_gb = 0.0
        swap_total_gb = swap_free_gb = 0.0

    if platform.system().lower() == "linux":
        cpu_model = get_cpu_model_linux()
        cpu_flags = get_cpu_flags_linux()
        cpu_mhz = get_cpu_mhz_linux()
    else:
        cpu_model = platform.processor() or "Unknown CPU"
        cpu_flags = []
        cpu_mhz = None

    return MachineInfo(
        os=os_name,
        arch=arch,
        cpu_model=cpu_model,
        physical_cores=physical,
        logical_cores=logical,
        cpu_mhz=cpu_mhz,
        cpu_flags=cpu_flags,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        swap_total_gb=swap_total_gb,
        swap_free_gb=swap_free_gb,
    )


# -----------------------------
# Memory estimation
# -----------------------------
@dataclass
class FitResult:
    family: str
    model: str
    task: str
    params_b: float
    quant: str
    bpw: float
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_required_gb: float
    fit: str
    notes: str


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def estimate_weights_gb(params_b: float, quant: str) -> tuple[float, float]:
    bpw = BPW_MAP.get(quant.upper(), 4.65)  # default conservador
    params = params_b * 1e9
    bytes_weights = params * (bpw / 8.0)
    gb = bytes_weights / (1024**3)
    return gb, bpw


def estimate_kv_cache_gb(spec: ModelSpec, ctx: int, kv_dtype_bytes: int, conservative_factor: float) -> float:
    """
    KV cache ~ 2 (K+V) * n_layers * ctx * hidden * bytes
    Usamos hints (layers/hidden) si están; si no, fallback: proporcional a params.
    """
    if spec.n_layers and spec.hidden:
        bytes_kv = 2.0 * spec.n_layers * ctx * spec.hidden * kv_dtype_bytes
        gb = (bytes_kv / (1024**3)) * conservative_factor
        return gb

    # Fallback: ~ 0.22 GB por 1B params a ctx=2048 (muy aproximado, conservador)
    gb = 0.22 * spec.params_b * (ctx / 2048.0) * conservative_factor
    return gb


def estimate_overhead_gb(weights_gb: float) -> float:
    # Overhead base + 10% del peso (buffers, tokenization, mmap, etc.)
    return 0.6 + 0.10 * weights_gb


def classify_fit(total_req: float, budget: float) -> str:
    # thresholds conservadores
    if total_req <= budget * 0.60:
        return "PERFECTO"
    if total_req <= budget * 0.80:
        return "BIEN"
    if total_req <= budget * 0.95:
        return "AJUSTADO"
    return "MAL"


def cpu_capability_hint(flags: list[str]) -> str:
    f = set(flags)
    if "avx512f" in f:
        return "CPU: AVX-512 (muy bien)"
    if "avx2" in f:
        return "CPU: AVX2 (bien)"
    if "avx" in f:
        return "CPU: AVX (ok)"
    return "CPU: sin AVX (lento; prioriza modelos <=3B)"


# -----------------------------
# Main scoring / ranking
# -----------------------------
def score_result(r: FitResult, preference: str) -> float:
    # Menor total requerido es mejor, y "fit" mejor es mejor.
    fit_bonus = {"PERFECTO": 0.0, "BIEN": 5.0, "AJUSTADO": 15.0, "MAL": 100.0}[r.fit]
    pref_bonus = 0.0
    if preference and preference != "any":
        if r.task == preference:
            pref_bonus -= 3.0
        # coding prefer: penaliza no-coding levemente
        elif preference == "coding" and r.task != "coding" or preference == "reasoning" and r.task != "reasoning":
            pref_bonus += 2.0

    # objetivo: minimizar (mem + penalizaciones)
    return r.total_required_gb + fit_bonus + pref_bonus


def build_recommendations(
    mi: MachineInfo,
    ctx: int,
    headroom: float,
    quants: list[str],
    kv_dtype_bytes: int,
    kv_conservative_factor: float,
    preference: str,
    top_n: int,
) -> tuple[list[FitResult], dict]:
    budget = max(0.0, mi.ram_total_gb * (1.0 - headroom))

    results: list[FitResult] = []
    for spec in MODEL_CATALOG:
        for q in quants:
            w_gb, bpw = estimate_weights_gb(spec.params_b, q)
            kv_gb = estimate_kv_cache_gb(
                spec, ctx=ctx, kv_dtype_bytes=kv_dtype_bytes, conservative_factor=kv_conservative_factor
            )
            ov_gb = estimate_overhead_gb(w_gb)
            total = w_gb + kv_gb + ov_gb
            fit = classify_fit(total, budget)
            results.append(
                FitResult(
                    family=spec.family,
                    model=spec.name,
                    task=spec.task,
                    params_b=spec.params_b,
                    quant=q.upper(),
                    bpw=bpw,
                    weights_gb=w_gb,
                    kv_cache_gb=kv_gb,
                    overhead_gb=ov_gb,
                    total_required_gb=total,
                    fit=fit,
                    notes=spec.notes,
                )
            )

    # rank
    results.sort(key=lambda r: score_result(r, preference))
    top = results[:top_n]

    summary = {
        "machine": asdict(mi),
        "budget_gb": budget,
        "ctx": ctx,
        "headroom": headroom,
        "kv_dtype_bytes": kv_dtype_bytes,
        "kv_conservative_factor": kv_conservative_factor,
        "preference": preference,
        "cpu_hint": cpu_capability_hint(mi.cpu_flags),
    }
    return top, summary


def print_table(results: list[FitResult]) -> None:
    rows = []
    for r in results:
        rows.append([
            r.fit,
            f"{r.model}",
            r.task,
            f"{r.params_b:.1f}B",
            r.quant,
            f"{r.total_required_gb:.2f} GB",
            f"W:{r.weights_gb:.2f}  KV:{r.kv_cache_gb:.2f}  OH:{r.overhead_gb:.2f}",
            r.notes,
        ])

    headers = ["FIT", "MODELO", "TAREA", "PARAMS", "QUANT", "RAM EST.", "DESGLOSE", "NOTAS"]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        # Fallback simple
        print(" | ".join(headers))
        print("-" * 120)
        for row in rows:
            print(" | ".join(str(x) for x in row))


def main() -> None:
    p = argparse.ArgumentParser(description="Analizador de máquina -> recomendador de modelos GGUF")
    p.add_argument("--ctx", type=int, default=2048, help="Context length (KV cache escala con esto). Default 2048.")
    p.add_argument("--headroom", type=float, default=0.25, help="Reserva de RAM (0.25 = usar 75%%).")
    p.add_argument("--quants", type=str, default=",".join(DEFAULT_QUANTS),
                   help="Lista de quants separados por coma. Ej: Q4_K_M,Q5_K_M,Q8_0")
    p.add_argument("--kv-bytes", type=int, default=2, help="Bytes por elemento para KV cache (2 = fp16).")
    p.add_argument("--kv-factor", type=float, default=1.20,
                   help="Factor conservador KV (>=1). Default 1.20 (sube un poco para evitar OOM).")
    p.add_argument("--prefer", type=str, default="any",
                   choices=["any", "general", "coding", "reasoning", "multilingual", "vision"],
                   help="Preferencia de tarea para ranking.")
    p.add_argument("--top", type=int, default=25, help="Top N recomendaciones.")
    p.add_argument("--json-out", type=str, default="", help="Ruta para exportar JSON (opcional).")
    args = p.parse_args()

    mi = get_machine_info()

    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    top, summary = build_recommendations(
        mi=mi,
        ctx=args.ctx,
        headroom=clamp(args.headroom, 0.05, 0.60),
        quants=quants,
        kv_dtype_bytes=args.kv_bytes,
        kv_conservative_factor=max(1.0, args.kv_factor),
        preference=args.prefer,
        top_n=max(5, args.top),
    )

    # Header
    print("=" * 88)
    print("LLM Machine Advisor (GGUF) - resultado")
    print("=" * 88)
    print(f"OS/ARCH : {mi.os} / {mi.arch}")
    print(f"CPU     : {mi.cpu_model}")
    print(f"CORES   : {mi.physical_cores} físicos / {mi.logical_cores} hilos")
    if mi.cpu_mhz:
        print(f"CPU MHz : ~{mi.cpu_mhz:.0f} MHz")
    print(f"RAM     : {mi.ram_total_gb:.2f} GB total | {mi.ram_available_gb:.2f} GB available")
    print(f"SWAP    : {mi.swap_total_gb:.2f} GB total | {mi.swap_free_gb:.2f} GB free")
    print(f"HINT    : {summary['cpu_hint']}")
    print(f"BUDGET  : {summary['budget_gb']:.2f} GB (headroom={summary['headroom']:.2f})")
    print(
        f"CTX     : {summary['ctx']} (KV bytes={summary['kv_dtype_bytes']}, "
        f"KV factor={summary['kv_conservative_factor']})"
    )
    print("=" * 88)
    print()

    print_table(top)
    print()

    # Warnings / ops hints
    if "avx2" not in set(mi.cpu_flags) and "avx512f" not in set(mi.cpu_flags):
        print("[WARN] CPU sin AVX2/AVX512: prioriza modelos <= 3B (Q4_K_M).")
    if mi.swap_total_gb > 0 and mi.swap_free_gb < (0.25 * mi.swap_total_gb):
        print("[WARN] Swap muy usada: riesgo de thrashing si corres modelos AJUSTADO.")
    if mi.ram_available_gb < (0.20 * mi.ram_total_gb):
        print("[WARN] RAM available baja ahora mismo: cierra procesos antes de probar modelos.")

    if args.json_out:
        payload = {
            "summary": summary,
            "recommendations": [asdict(r) for r in top],
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] JSON exportado en: {args.json_out}")


if __name__ == "__main__":
    main()
