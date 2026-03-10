from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class ExcelBanditData:
    df: pd.DataFrame
    source_col: str
    context_cols: List[str]
    treatment_col: str
    control_col: str
    train_count_col: str
    test_count_col: str
    total_count_col: Optional[str]

    contexts: torch.Tensor
    treatment: torch.Tensor
    control: torch.Tensor
    train_counts: torch.Tensor
    test_counts: torch.Tensor

    source_values: List[Any]
    source_indices: List[torch.Tensor]
    source_train_probs: List[torch.Tensor]
    source_test_probs: List[torch.Tensor]

    pooled_train_probs: torch.Tensor
    pooled_test_probs: torch.Tensor

    inner_contexts: torch.Tensor
    inner_treatment: torch.Tensor
    inner_control: torch.Tensor

    context_scale: str

    @property
    def n_sources(self) -> int:
        return len(self.source_values)

    @property
    def d_context(self) -> int:
        return int(self.contexts.shape[1])


def _as_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.factorize(out[c])[0].astype(float)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _scale_contexts(ctx: np.ndarray, mode: str) -> np.ndarray:
    x = np.array(ctx, dtype=float, copy=True)
    if mode == "none":
        return x

    if mode == "minmax":
        lo = x.min(axis=0, keepdims=True)
        hi = x.max(axis=0, keepdims=True)
        den = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
        return (x - lo) / den

    if mode == "zscore":
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        den = np.where(sd < 1e-12, 1.0, sd)
        return (x - mu) / den

    raise ValueError(f"Unknown context scaling mode: {mode}")


def _normalize(weights: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(weights, min=0.0)
    s = w.sum()
    if s <= 0:
        return torch.ones_like(w) / float(len(w))
    return w / s


def _build_inner_pool(
    contexts: torch.Tensor,
    treatment: torch.Tensor,
    control: torch.Tensor,
    train_counts: torch.Tensor,
    use_unique_contexts: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_unique_contexts:
        return contexts, treatment, control

    # Deduplicate numerically close contexts and aggregate costs by train-count weight.
    x = contexts.detach().cpu().numpy()
    key = np.round(x, decimals=12)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)

    treat_u = np.zeros(len(uniq), dtype=float)
    ctrl_u = np.zeros(len(uniq), dtype=float)
    w_u = np.zeros(len(uniq), dtype=float)

    t_np = treatment.detach().cpu().numpy()
    c_np = control.detach().cpu().numpy()
    w_np = train_counts.detach().cpu().numpy()

    for i, j in enumerate(inv):
        w = float(max(w_np[i], 0.0))
        w_u[j] += w
        treat_u[j] += w * float(t_np[i])
        ctrl_u[j] += w * float(c_np[i])

    w_u = np.where(w_u <= 0.0, 1.0, w_u)
    treat_u = treat_u / w_u
    ctrl_u = ctrl_u / w_u

    dtype = contexts.dtype
    device = contexts.device
    return (
        torch.as_tensor(uniq, dtype=dtype, device=device),
        torch.as_tensor(treat_u, dtype=dtype, device=device),
        torch.as_tensor(ctrl_u, dtype=dtype, device=device),
    )


def load_excel_bandit(
    data_path: str,
    source_col: str,
    context_cols: Optional[List[str]],
    treatment_col: str,
    control_col: str,
    train_count_col: str,
    test_count_col: str,
    total_count_col: Optional[str] = None,
    source_values: Optional[List[Any]] = None,
    context_scale: str = "zscore",
    use_unique_contexts: bool = True,
    dtype: torch.dtype = torch.float64,
) -> ExcelBanditData:
    df = pd.read_excel(data_path).copy()

    required = [source_col, treatment_col, control_col, train_count_col, test_count_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if context_cols is None or len(context_cols) == 0:
        reserved = {
            source_col,
            treatment_col,
            control_col,
            train_count_col,
            test_count_col,
        }
        if total_count_col is not None:
            reserved.add(total_count_col)
        context_cols = [c for c in df.columns if c not in reserved]

    missing_ctx = [c for c in context_cols if c not in df.columns]
    if missing_ctx:
        raise ValueError(f"Missing context columns: {missing_ctx}")

    if source_values is not None and len(source_values) > 0:
        keep = {str(v) for v in source_values}
        df = df[df[source_col].astype(str).isin(keep)].copy()
        if df.empty:
            raise ValueError("Filtering by source_values removed all rows")

    df = _as_numeric_df(df, context_cols + [treatment_col, control_col, train_count_col, test_count_col])

    ctx_np = df[context_cols].to_numpy(dtype=float)
    ctx_np = _scale_contexts(ctx_np, context_scale)

    contexts = torch.as_tensor(ctx_np, dtype=dtype)
    treatment = torch.as_tensor(df[treatment_col].to_numpy(dtype=float), dtype=dtype)
    control = torch.as_tensor(df[control_col].to_numpy(dtype=float), dtype=dtype)
    train_counts = torch.as_tensor(df[train_count_col].to_numpy(dtype=float), dtype=dtype)
    test_counts = torch.as_tensor(df[test_count_col].to_numpy(dtype=float), dtype=dtype)

    pooled_train_probs = _normalize(train_counts)
    pooled_test_probs = _normalize(test_counts)

    observed_sources = list(df[source_col].drop_duplicates())
    if source_values is not None and len(source_values) > 0:
        observed_by_str = {str(v): v for v in observed_sources}
        source_values_final = [observed_by_str[str(v)] for v in source_values if str(v) in observed_by_str]
    else:
        source_values_final = observed_sources
    source_indices: List[torch.Tensor] = []
    source_train_probs: List[torch.Tensor] = []
    source_test_probs: List[torch.Tensor] = []

    source_arr = df[source_col].to_numpy()
    for s in source_values_final:
        idx_np = np.where(source_arr == s)[0]
        idx = torch.as_tensor(idx_np, dtype=torch.long)
        source_indices.append(idx)

        tr = train_counts[idx]
        te = test_counts[idx]
        source_train_probs.append(_normalize(tr))
        source_test_probs.append(_normalize(te))

    inner_contexts, inner_treatment, inner_control = _build_inner_pool(
        contexts=contexts,
        treatment=treatment,
        control=control,
        train_counts=train_counts,
        use_unique_contexts=use_unique_contexts,
    )

    return ExcelBanditData(
        df=df,
        source_col=source_col,
        context_cols=context_cols,
        treatment_col=treatment_col,
        control_col=control_col,
        train_count_col=train_count_col,
        test_count_col=test_count_col,
        total_count_col=total_count_col,
        contexts=contexts,
        treatment=treatment,
        control=control,
        train_counts=train_counts,
        test_counts=test_counts,
        source_values=source_values_final,
        source_indices=source_indices,
        source_train_probs=source_train_probs,
        source_test_probs=source_test_probs,
        pooled_train_probs=pooled_train_probs,
        pooled_test_probs=pooled_test_probs,
        inner_contexts=inner_contexts,
        inner_treatment=inner_treatment,
        inner_control=inner_control,
        context_scale=context_scale,
    )


def sample_outer_by_source(
    data: ExcelBanditData,
    source_idx: int,
    n: int,
    split: str = "train",
) -> torch.Tensor:
    idx_all = data.source_indices[source_idx]
    probs = data.source_train_probs[source_idx] if split == "train" else data.source_test_probs[source_idx]
    pick_local = torch.multinomial(probs, num_samples=n, replacement=True)
    pick_global = idx_all[pick_local]
    return data.contexts[pick_global]


def sample_outer_pooled(
    data: ExcelBanditData,
    n: int,
    split: str = "train",
) -> torch.Tensor:
    probs = data.pooled_train_probs if split == "train" else data.pooled_test_probs
    idx = torch.multinomial(probs, num_samples=n, replacement=True)
    return data.contexts[idx]


def sample_inner_uniform(data: ExcelBanditData, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = int(data.inner_contexts.shape[0])
    idx = torch.randint(low=0, high=m, size=(n,))
    return data.inner_contexts[idx], data.inner_treatment[idx], data.inner_control[idx]
