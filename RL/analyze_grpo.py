#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def analyze(
    csv_path,
    outdir="grpo_reward_analysis",
    # Grouping: GRPO is per micro-step; best default is (trigger, micro)
    group_cols=("trigger", "micro"),
    # Optional: specify target for tracking-error diagnostics
    target_pct=None,   # e.g., 0.25
):
    csv_path = Path(csv_path)
    outdir = Path(outdir) / csv_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"\n== {csv_path} ==")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    # Basic required columns
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Missing group col '{c}'. Available: {list(df.columns)}")

    # prefer phase if present
    has_phase = "phase" in df.columns
    if not has_phase and "executed" not in df.columns:
        raise ValueError("Need either 'phase' column or 'executed' (0/1) column.")

    # numeric conversions
    df = _to_num(df, [
        "reward_raw", "reward_exec", "reward_best_sample",
        "bg_before", "bg_after", "delta", "occ_mid",
        "tt_after", "aa_after",
        "executed", "shielded",
        "chunk", "micro", "micro_global", "k", "a",
    ])

    # -----------------------------
    # Split candidate vs executed
    # -----------------------------
    if has_phase:
        cand = df[df["phase"] == "candidate"].copy()
        execu = df[df["phase"] == "executed"].copy()
    else:
        executed = df["executed"].fillna(0).astype(int)
        execu = df[executed == 1].copy()
        cand = df[executed == 0].copy()

    # Decide reward columns
    if "reward_raw" not in df.columns:
        raise ValueError("Expected 'reward_raw' in this GRPO trace CSV for candidates.")
    if "reward_exec" not in df.columns and "reward_raw" not in df.columns:
        raise ValueError("Expected at least one of reward_exec or reward_raw.")

    # Candidate reward = reward_raw
    cand_reward_col = "reward_raw"
    # Executed reward = reward_exec if exists else reward_raw
    exec_reward_col = "reward_exec" if "reward_exec" in df.columns else "reward_raw"

    # -----------------------------
    # Global candidate reward stats
    # -----------------------------
    r = cand[cand_reward_col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    print(f"\nUsing candidate reward col: {cand_reward_col}")
    print("Candidate reward min/mean/max:", float(np.min(r)), float(np.mean(r)), float(np.max(r)))
    print("Candidate reward p1/p5/p50/p95/p99:", np.percentile(r, [1,5,50,95,99]).tolist())

    plt.figure()
    plt.hist(r, bins=80)
    plt.xlabel(cand_reward_col)
    plt.ylabel("count")
    plt.title("Candidate reward histogram (phase=candidate)")
    plt.savefig(outdir / "candidate_reward_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Sanity: candidate group size and executed rows per group
    # -----------------------------
    cand_counts = cand.groupby(list(group_cols)).size().rename("n_candidates").reset_index()
    exec_counts = execu.groupby(list(group_cols)).size().rename("n_executed").reset_index()

    sanity = cand_counts.merge(exec_counts, on=list(group_cols), how="outer").fillna(0)
    sanity.to_csv(outdir / "sanity_counts_per_group.csv", index=False)

    print("\nSanity (per group):")
    print("n_candidates min/median/max =",
          int(sanity["n_candidates"].min()),
          float(sanity["n_candidates"].median()),
          int(sanity["n_candidates"].max()))
    print("n_executed   min/median/max =",
          int(sanity["n_executed"].min()),
          float(sanity["n_executed"].median()),
          int(sanity["n_executed"].max()))

    # -----------------------------
    # Per-group candidate stats (mean/std/range)
    # -----------------------------
    stats = cand.groupby(list(group_cols))[cand_reward_col].agg(
        count="count", mean="mean", std="std", min="min", max="max"
    ).reset_index()
    stats["range"] = stats["max"] - stats["min"]
    stats.to_csv(outdir / "per_group_candidate_reward_stats.csv", index=False)

    # Plot mean ± std and range over time-like axis if present
    time_col = "micro_global" if "micro_global" in df.columns else (group_cols[-1] if len(group_cols) else None)

    if time_col and time_col in df.columns:
        # need a representative time per group (take min)
        tmap = df.groupby(list(group_cols))[time_col].min().reset_index().rename(columns={time_col: "_t"})
        stats_t = stats.merge(tmap, on=list(group_cols), how="left").sort_values("_t")

        plt.figure()
        plt.plot(stats_t["_t"], stats_t["mean"], label="mean")
        plt.fill_between(stats_t["_t"],
                         (stats_t["mean"] - stats_t["std"].fillna(0)),
                         (stats_t["mean"] + stats_t["std"].fillna(0)),
                         alpha=0.2)
        plt.xlabel(time_col); plt.ylabel("candidate reward")
        plt.title("Candidate reward mean ± std per micro-step group")
        plt.savefig(outdir / "candidate_reward_mean_std_vs_time.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(stats_t["_t"], stats_t["range"])
        plt.xlabel(time_col); plt.ylabel("max-min candidate reward")
        plt.title("Within-group reward range (informativeness of sampling)")
        plt.savefig(outdir / "candidate_reward_range_vs_time.png", dpi=200, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # Best sampled reward per group (from candidates)
    # -----------------------------
    best = cand.groupby(list(group_cols))[cand_reward_col].max().rename("r_best").reset_index()

    # -----------------------------
    # Executed reward per group
    # -----------------------------
    if len(execu) == 0:
        print("\nWARNING: no executed rows found; skipping executed-vs-best and shielding analyses.")
        exec_df = None
    else:
        execu["r_exec"] = execu[exec_reward_col]
        execu["r_exec"] = execu["r_exec"].where(np.isfinite(execu["r_exec"]), execu.get("reward_raw"))
        exec_df = execu.groupby(list(group_cols))["r_exec"].mean().reset_index()

    # -----------------------------
    # Regret / best-vs-executed gap
    # -----------------------------
    if exec_df is not None:
        merged = best.merge(exec_df, on=list(group_cols), how="inner")
        merged["gap_best_minus_exec"] = merged["r_best"] - merged["r_exec"]
        merged.to_csv(outdir / "best_vs_exec_gap.csv", index=False)

        # attach time if possible
        if time_col and time_col in df.columns:
            tmap = df.groupby(list(group_cols))[time_col].min().reset_index().rename(columns={time_col: "_t"})
            merged = merged.merge(tmap, on=list(group_cols), how="left").sort_values("_t")
            x = merged["_t"].to_numpy()
            xlabel = time_col
        else:
            x = np.arange(len(merged))
            xlabel = "group index"

        plt.figure()
        plt.plot(x, merged["r_exec"], label="executed")
        plt.plot(x, merged["r_best"], label="best sampled (candidate max)")
        plt.xlabel(xlabel); plt.ylabel("reward")
        plt.title("Executed vs best-sampled reward (per micro-step group)")
        plt.legend()
        plt.savefig(outdir / "executed_vs_best.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(x, merged["gap_best_minus_exec"])
        plt.xlabel(xlabel); plt.ylabel("best - executed")
        plt.title("Gap (want small; large often means shielding or noise)")
        plt.savefig(outdir / "gap_best_minus_exec.png", dpi=200, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # Action coverage (candidate vs executed)
    # -----------------------------
    if "a" in df.columns:
        plt.figure()
        cand["a"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("action a"); plt.ylabel("count (candidate samples)")
        plt.title("Candidate action histogram")
        plt.savefig(outdir / "candidate_action_hist.png", dpi=200, bbox_inches="tight")
        plt.close()

        if len(execu) > 0:
            plt.figure()
            execu["a"].value_counts().sort_index().plot(kind="bar")
            plt.xlabel("action a"); plt.ylabel("count (executed)")
            plt.title("Executed action histogram")
            plt.savefig(outdir / "executed_action_hist.png", dpi=200, bbox_inches="tight")
            plt.close()

    # -----------------------------
    # Shielding analysis
    # -----------------------------
    if "shielded" in df.columns and len(execu) > 0:
        sh = execu.groupby(list(group_cols))["shielded"].mean().reset_index().rename(columns={"shielded": "shield_rate"})
        sh.to_csv(outdir / "shield_rate_per_group.csv", index=False)

        if time_col and time_col in df.columns:
            tmap = df.groupby(list(group_cols))[time_col].min().reset_index().rename(columns={time_col: "_t"})
            sh = sh.merge(tmap, on=list(group_cols), how="left").sort_values("_t")
            x = sh["_t"].to_numpy()
            xlabel = time_col
        else:
            x = np.arange(len(sh))
            xlabel = "group index"

        plt.figure()
        plt.plot(x, sh["shield_rate"])
        plt.xlabel(xlabel); plt.ylabel("mean(shielded)")
        plt.title("Shielding rate over time (executed rows)")
        plt.savefig(outdir / "shield_rate_vs_time.png", dpi=200, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # Optional: tracking-error diagnostics if target_pct provided
    # -----------------------------
    if target_pct is not None and "bg_after" in cand.columns:
        cand["abs_bg_err"] = np.abs(cand["bg_after"] - float(target_pct))
        # correlation table
        cols = ["reward_raw", "abs_bg_err"]
        for extra in ["occ_mid", "delta", "tt_after", "aa_after"]:
            if extra in cand.columns:
                cols.append(extra)
        corr = cand[cols].corr(numeric_only=True)["reward_raw"].sort_values(ascending=False)
        corr.to_csv(outdir / "candidate_reward_correlations.csv")
        print("\nCandidate reward correlations (saved):")
        print(corr)
    
    # -----------------------------
    # Extra diagnostics
    # -----------------------------
    # In-band mask for candidates (uses bg_after relative to target_pct and inferred tol if available)
    if target_pct is not None and "bg_after" in cand.columns:
        # If you know tol, set it here; otherwise just visualize abs error
        cand["abs_bg_err"] = np.abs(cand["bg_after"] - float(target_pct))

        # reward vs abs_bg_err
        plt.figure()
        plt.scatter(cand["abs_bg_err"], cand["reward_raw"], s=3, alpha=0.3)
        plt.xlabel("|bg_after - target| (percent units)")
        plt.ylabel("reward_raw")
        plt.title("Reward vs tracking error (candidates)")
        plt.savefig(outdir / "scatter_reward_vs_abs_bg_err.png", dpi=200, bbox_inches="tight")
        plt.close()

        # reward vs tt/aa (only if present)
        for col in ["tt_after", "aa_after"]:
            if col in cand.columns:
                plt.figure()
                plt.scatter(cand[col], cand["reward_raw"], s=3, alpha=0.3)
                plt.xlabel(col)
                plt.ylabel("reward_raw")
                plt.title(f"Reward vs {col} (candidates)")
                plt.savefig(outdir / f"scatter_reward_vs_{col}.png", dpi=200, bbox_inches="tight")
                plt.close()

    # Gap stats + shielding effect (executed rows)
    if exec_df is not None and len(execu) > 0:
        merged2 = best.merge(exec_df, on=list(group_cols), how="inner")
        merged2["gap"] = merged2["r_best"] - merged2["r_exec"]

        # attach shield info (mean shielded per group from executed rows)
        if "shielded" in execu.columns:
            shg = execu.groupby(list(group_cols))["shielded"].mean().reset_index()
            merged2 = merged2.merge(shg, on=list(group_cols), how="left")

        merged2.to_csv(outdir / "gap_with_shielding.csv", index=False)

        print("\nGap(best-exec) summary:",
              "mean=", float(np.mean(merged2["gap"])),
              "p50=", float(np.percentile(merged2["gap"], 50)),
              "p95=", float(np.percentile(merged2["gap"], 95)))

        if "shielded" in merged2.columns:
            g0 = merged2[merged2["shielded"] < 0.5]["gap"]
            g1 = merged2[merged2["shielded"] >= 0.5]["gap"]
            if len(g0) and len(g1):
                print("Gap mean when shielded=0:", float(np.mean(g0)),
                      "| shielded=1:", float(np.mean(g1)))

        # histogram of gap
        plt.figure()
        plt.hist(merged2["gap"], bins=80)
        plt.xlabel("gap = best_sampled - executed")
        plt.ylabel("count")
        plt.title("Gap histogram (per micro-step)")
        plt.savefig(outdir / "gap_hist.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Per-trigger plots (candidate reward hist)
    if "trigger" in cand.columns:
        for tr, sub in cand.groupby("trigger"):
            rr = sub["reward_raw"].to_numpy(dtype=float)
            rr = rr[np.isfinite(rr)]
            plt.figure()
            plt.hist(rr, bins=80)
            plt.xlabel("reward_raw")
            plt.ylabel("count")
            plt.title(f"Candidate reward histogram ({tr})")
            plt.savefig(outdir / f"candidate_reward_hist_{tr}.png", dpi=200, bbox_inches="tight")
            plt.close()


    # -----------------------------
    # Split-by-trigger summary
    # -----------------------------
    if "trigger" in df.columns:
        trig_summary = []
        for tr, sub in cand.groupby("trigger"):
            rr = sub["reward_raw"].to_numpy(dtype=float)
            rr = rr[np.isfinite(rr)]
            trig_summary.append({
                "trigger": tr,
                "n_candidate_rows": len(sub),
                "reward_mean": float(np.mean(rr)) if len(rr) else np.nan,
                "reward_std": float(np.std(rr)) if len(rr) else np.nan,
                "reward_p95": float(np.percentile(rr, 95)) if len(rr) else np.nan,
            })
        pd.DataFrame(trig_summary).to_csv(outdir / "candidate_reward_summary_by_trigger.csv", index=False)

    print("\nWrote outputs to:", outdir)

if __name__ == "__main__":
    analyze("outputs/demo_sing_grpo_as_feature/grpo_as_ht_sampled_rewards.csv",
            target_pct=0.25,
            group_cols=("trigger","micro"))
