import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoLingo Dashboard",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fc;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        border-left: 5px solid #667eea;
        margin-bottom: 0.7rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa, #e8ecf8);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        border: 2px solid #d0d8f0;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .pred-label { font-size: 0.82rem; color: #777; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
    .pred-char  { font-size: 3.2rem; font-weight: 900; color: #3a3aff; font-family: monospace; }
    .pred-word  { font-size: 1.3rem; font-weight: 700; color: #333; font-family: monospace; margin-top: 0.3rem; }
    .context-box { font-size: 1.4rem; font-family: monospace; background: #1e1e2e; color: #cdd6f4;
                   border-radius: 10px; padding: 0.8rem 1.2rem; letter-spacing: 0.08em; }
    .stTextInput > div > div > input { font-size: 1.2rem; font-family: monospace; border-radius: 10px; }
    .how-it-works { background: #f0f4ff; border-radius: 12px; padding: 1rem 1.4rem; border: 1px solid #c8d4f8; }
    .topk-bar-bg  { background: #e8ecf8; border-radius: 6px; height: 18px; margin: 3px 0; }
    .topk-bar-fill{ background: linear-gradient(90deg, #667eea, #764ba2);
                    border-radius: 6px; height: 18px; transition: width 0.3s; }
    .cache-badge  { background: #d4edda; color: #155724; border-radius: 6px;
                    padding: 0.25rem 0.6rem; font-size: 0.78rem; font-weight: 600; }
    .train-badge  { background: #fff3cd; color: #856404; border-radius: 6px;
                    padding: 0.25rem 0.6rem; font-size: 0.78rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "logs/sibling_b.csv"
BENCH_PATH   = "results/benchmark_results.json"
METRICS_PATH = "logs/metrics_log.csv"
RUN_LOGS     = "run_logs.csv"
MODELS_DIR   = "models"
MODEL_CACHE  = os.path.join(MODELS_DIR, "autolingo_models.joblib")

# Colour map for all 5 models
MODEL_COLORS = {
    "KNN":               "#4f8ef7",
    "Decision Tree":     "#f76b4f",
    "Label Propagation": "#4fc97f",
    "Logistic Reg":      "#f7c44f",
    "Random Forest":     "#a374e8",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING / LOADING  (cached at process level)
# ─────────────────────────────────────────────────────────────────────────────
def _build_and_save():
    """Train all 5 classifiers and persist them with joblib."""
    df = pd.read_csv(DATA_PATH)

    all_chars = set()
    for col in df.columns:
        all_chars.update(df[col].astype(str).unique())
    vocab      = sorted(all_chars)
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}

    c1 = df["Readable_Char1"].astype(str).map(char_to_id).fillna(0).astype(int)
    c2 = df["Readable_Char2"].astype(str).map(char_to_id).fillna(0).astype(int)
    c3 = df["Readable_Char3"].astype(str).map(char_to_id).fillna(0).astype(int)
    X  = np.stack([c1, c2, c3], axis=1)
    y  = df["Readable_Actual_Target"].astype(str).map(char_to_id).fillna(0).astype(int)

    # ── KNN ──────────────────────────────────────────────────────────────────
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X, y)

    # ── Decision Tree ─────────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=15, random_state=42)
    dt.fit(X, y)

    # ── Label Propagation (semi-supervised) ───────────────────────────────────
    y_semi = y.copy()
    rng    = np.random.default_rng(42)
    mask   = rng.random(len(y_semi)) < 0.30
    y_semi[mask] = -1
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=300)
    lp.fit(X, y_semi)

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=500, solver="lbfgs",
                            random_state=42, C=1.0)
    lr.fit(X, y)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # ── Persist to disk ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        dict(knn=knn, dt=dt, lp=lp, lr=lr, rf=rf,
             char_to_id=char_to_id, id_to_char=id_to_char, vocab=vocab),
        MODEL_CACHE
    )
    return knn, dt, lp, lr, rf, char_to_id, id_to_char, vocab, False  # False = freshly trained


@st.cache_resource(show_spinner="⏳ Loading / training ML models…")
def load_and_train(force_retrain: bool = False):
    """
    Load models from disk if available, otherwise train from scratch.
    Pass force_retrain=True (by bumping a counter in session_state) to
    bypass the cache and retrain.
    """
    if not force_retrain and os.path.exists(MODEL_CACHE):
        try:
            data = joblib.load(MODEL_CACHE)
            required = {"knn", "dt", "lp", "lr", "rf", "char_to_id", "id_to_char", "vocab"}
            if required.issubset(data.keys()):
                return (data["knn"], data["dt"], data["lp"], data["lr"], data["rf"],
                        data["char_to_id"], data["id_to_char"], data["vocab"], True)  # True = from cache
        except Exception:
            pass

    return _build_and_save()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _build_feature(prompt: str, char_to_id: dict) -> np.ndarray:
    """Return a (1, 3) feature array for the last 3 chars of prompt."""
    if len(prompt) < 3:
        prompt = " " * (3 - len(prompt)) + prompt
    c1 = char_to_id.get(prompt[-3], 0)
    c2 = char_to_id.get(prompt[-2], 0)
    c3 = char_to_id.get(prompt[-1], 0)
    return np.array([[c1, c2, c3]])


def predict_next(prompt: str, knn, dt, lp, lr, rf, char_to_id, id_to_char) -> dict:
    """Predict the single best next character for each of the 5 models."""
    X_new = _build_feature(prompt, char_to_id)

    def safe_predict(model):
        try:
            return id_to_char.get(int(model.predict(X_new)[0]), "?")
        except Exception:
            return "?"

    return {
        "KNN":               safe_predict(knn),
        "Decision Tree":     safe_predict(dt),
        "Label Propagation": safe_predict(lp),
        "Logistic Reg":      safe_predict(lr),
        "Random Forest":     safe_predict(rf),
    }


def predict_topk(prompt: str, model, char_to_id: dict, id_to_char: dict, k: int = 5) -> list:
    """
    Return top-K (character, probability) pairs using predict_proba.
    Falls back to [(predicted_char, 1.0)] for models without proba support.
    """
    X_new = _build_feature(prompt, char_to_id)
    try:
        proba   = model.predict_proba(X_new)[0]       # shape: (n_classes,)
        classes = model.classes_                        # class IDs in same order
        pairs   = [(id_to_char.get(int(cls), "?"), float(p))
                   for cls, p in zip(classes, proba)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]
    except Exception:
        try:
            pred = id_to_char.get(int(model.predict(X_new)[0]), "?")
            return [(pred, 1.0)]
        except Exception:
            return [("?", 1.0)]


def generate_completion(prompt: str, model_name: str,
                        knn, dt, lp, lr, rf,
                        char_to_id, id_to_char, steps: int = 20) -> str:
    """Auto-complete prompt by repeatedly predicting the next character."""
    model_map = {
        "KNN":               knn,
        "Decision Tree":     dt,
        "Label Propagation": lp,
        "Logistic Reg":      lr,
        "Random Forest":     rf,
    }
    model  = model_map[model_name]
    result = prompt

    for _ in range(steps):
        ctx = result if len(result) >= 3 else " " * (3 - len(result)) + result
        c1 = char_to_id.get(ctx[-3], 0)
        c2 = char_to_id.get(ctx[-2], 0)
        c3 = char_to_id.get(ctx[-1], 0)
        try:
            pred_id = model.predict(np.array([[c1, c2, c3]]))[0]
            next_ch = id_to_char.get(int(pred_id), " ")
        except Exception:
            next_ch = " "
        result += next_ch
        if next_ch == "\n":
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/keyboard.png", width=70)
    st.markdown("## AutoLingo")
    st.markdown("*AI Smart Text Autocomplete*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Live Autocomplete", "Benchmark Results", "Performance Forecaster", "How It Works"],
        index=0
    )

    st.divider()

    # ── Model cache controls ──────────────────────────────────────────────────
    st.markdown("### 💾 Model Cache")

    cache_exists = os.path.exists(MODEL_CACHE)
    if cache_exists:
        mtime = os.path.getmtime(MODEL_CACHE)
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
        st.markdown(f'<span class="cache-badge">✅ Cached — {ts}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="train-badge">⚙️ Will train fresh</span>', unsafe_allow_html=True)

    if st.button("🔄 Retrain & overwrite cache", use_container_width=True):
        if os.path.exists(MODEL_CACHE):
            os.remove(MODEL_CACHE)
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("**Team Member 4** — Model Evaluator")
    st.markdown("`final project.ipynb`")
    st.caption("AutoLingo Intern Project")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
# Use a session key to force re-runs when retrain is requested
force = st.session_state.get("force_retrain", False)
result_tuple = load_and_train(force_retrain=force)
knn, dt, lp, lr, rf, char_to_id, id_to_char, vocab, from_cache = result_tuple

# Show a banner based on model source
if from_cache:
    st.success("💾 Models loaded from disk — no retraining was needed.")
else:
    st.info("⚙️ Models freshly trained and saved to `models/autolingo_models.joblib`.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: LIVE AUTOCOMPLETE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Live Autocomplete":
    st.markdown('<div class="main-title">AutoLingo ⌨️</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Type any text — watch all <strong>5 ML models</strong> '
        'predict the next character live</div>',
        unsafe_allow_html=True
    )

    col_input, col_settings = st.columns([3, 1])
    with col_input:
        prompt = st.text_input(
            "Your prompt",
            value="hello",
            placeholder="Type something here…",
            label_visibility="collapsed"
        )
    with col_settings:
        complete_steps = st.slider("Completion length", 5, 50, 20, 5)
        topk_k = st.slider("Top-K candidates", 3, 10, 5, 1)

    if prompt:
        # ── Single next-character predictions ────────────────────────────────
        preds = predict_next(prompt, knn, dt, lp, lr, rf, char_to_id, id_to_char)

        st.markdown("---")
        st.markdown(
            f"#### Context window: "
            f"<span class='context-box'>{prompt[-3:] if len(prompt) >= 3 else prompt}</span>",
            unsafe_allow_html=True
        )
        st.markdown("##### Next character predicted by each model:")

        cols = st.columns(5)
        for col, (name, char) in zip(cols, preds.items()):
            color = MODEL_COLORS[name]
            with col:
                st.markdown(f"""
                <div class="prediction-box" style="border-color:{color};">
                    <div class="pred-label">{name}</div>
                    <div class="pred-char" style="color:{color};">{char if char.strip() else '⎵'}</div>
                    <div style="font-size:0.8rem; color:#999; margin-top:4px;">next character</div>
                </div>""", unsafe_allow_html=True)

        # ── Agreement indicator ───────────────────────────────────────────────
        unique_preds = set(preds.values())
        if len(unique_preds) == 1:
            st.success(f"✅ All 5 models agree: next character = **'{list(unique_preds)[0]}'**")
        elif len(unique_preds) <= 2:
            st.warning(f"⚠️ Models mostly agree — {6 - len(unique_preds)} out of 5 predict the same character.")
        else:
            st.info(f"ℹ️ {len(unique_preds)} different predictions — high uncertainty.")

        # ── TOP-K CANDIDATES ─────────────────────────────────────────────────
        st.markdown("---")
        with st.expander(f"🔢 Top-{topk_k} next-character candidates (with probabilities)", expanded=True):
            st.caption(
                "Models with `predict_proba()` support: KNN, Decision Tree, "
                "Logistic Reg, Random Forest. Label Propagation uses its label distribution."
            )

            topk_cols = st.columns(5)
            model_objs = {
                "KNN": knn, "Decision Tree": dt, "Label Propagation": lp,
                "Logistic Reg": lr, "Random Forest": rf
            }

            for col, (mname, mobj) in zip(topk_cols, model_objs.items()):
                color = MODEL_COLORS[mname]
                topk  = predict_topk(prompt, mobj, char_to_id, id_to_char, k=topk_k)

                with col:
                    st.markdown(f"**{mname}**")
                    rows = []
                    for rank, (ch, prob) in enumerate(topk, 1):
                        display_ch = "⎵" if ch == " " else ch
                        bar_w = int(prob * 100)
                        rows.append({
                            "Rank": rank,
                            "Char": display_ch,
                            "Prob": f"{prob:.3f}",
                        })
                        # Mini inline bar
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
                            f'  <span style="font-family:monospace;font-weight:700;'
                            f'        font-size:1.1rem;color:{color};width:22px;">{display_ch}</span>'
                            f'  <div class="topk-bar-bg" style="flex:1;">'
                            f'    <div class="topk-bar-fill" style="width:{bar_w}%;'
                            f'         background:{color};opacity:0.8;"></div>'
                            f'  </div>'
                            f'  <span style="font-size:0.75rem;color:#666;width:40px;">{prob:.3f}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        # ── FULL AUTO-COMPLETION ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### Full auto-completion:")

        tab_knn, tab_dt, tab_lp, tab_lr, tab_rf = st.tabs([
            "KNN Classifier", "Decision Tree", "Label Propagation",
            "Logistic Reg", "Random Forest"
        ])

        for tab, mname in zip(
            [tab_knn, tab_dt, tab_lp, tab_lr, tab_rf],
            ["KNN", "Decision Tree", "Label Propagation", "Logistic Reg", "Random Forest"]
        ):
            with tab:
                completed = generate_completion(
                    prompt, mname, knn, dt, lp, lr, rf,
                    char_to_id, id_to_char, steps=complete_steps
                )
                original  = completed[:len(prompt)]
                extension = completed[len(prompt):]
                st.code(f"{original}[{extension}]", language="text")
                st.caption(f"[ ] = model-generated portion    |    {mname} model")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: BENCHMARK RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Benchmark Results":
    st.markdown('<div class="main-title">Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Accuracy &amp; F1 comparison across all models</div>', unsafe_allow_html=True)

    if os.path.exists(BENCH_PATH):
        with open(BENCH_PATH) as f:
            bench = json.load(f)

        df_bench = pd.DataFrame(bench)
        df_bench.columns = ["Model", "Accuracy", "F1 Score"]
        df_bench = df_bench.sort_values("Accuracy", ascending=False).reset_index(drop=True)

        col_table, col_bar = st.columns([1, 1.5])

        with col_table:
            st.markdown("#### Score Table")
            st.dataframe(
                df_bench.style
                    .format({"Accuracy": "{:.4f}", "F1 Score": "{:.4f}"})
                    .background_gradient(subset=["Accuracy", "F1 Score"], cmap="Blues"),
                use_container_width=True
            )
            best = df_bench.iloc[0]
            st.success(f"Best model: **{best['Model']}** with Accuracy = {best['Accuracy']:.4f}")

        with col_bar:
            st.markdown("#### Visual Comparison")
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            colors = ["#4f8ef7", "#f76b4f", "#4fc97f", "#a374e8"]

            for ax, metric in zip(axes, ["Accuracy", "F1 Score"]):
                bars = ax.bar(df_bench["Model"], df_bench[metric], color=colors[:len(df_bench)])
                ax.set_title(metric, fontweight="bold")
                ax.set_ylim(0, max(df_bench[metric].max() * 1.2, 0.5))
                ax.tick_params(axis='x', rotation=30)
                for bar, val in zip(bars, df_bench[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
                ax.grid(axis='y', alpha=0.3)
                ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Training loss chart ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Neural Network Training Loss Curve")

    if os.path.exists(METRICS_PATH):
        df_metrics = pd.read_csv(METRICS_PATH)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df_metrics["step"], df_metrics["train_loss"], label="Train Loss", color="#4f8ef7", alpha=0.8)
        ax2.plot(df_metrics["step"], df_metrics["val_loss"],   label="Val Loss",   color="#f76b4f", alpha=0.8)
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Loss")
        ax2.set_title("PyTorch Neural Net — Training Progress")
        ax2.legend()
        ax2.grid(True, alpha=0.25)
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Final Val Loss",  f"{df_metrics['val_loss'].iloc[-1]:.4f}")
        col_m2.metric("Best Val Loss",   f"{df_metrics['val_loss'].min():.4f}")
        col_m3.metric("Total Steps",     f"{int(df_metrics['step'].iloc[-1]):,}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: PERFORMANCE FORECASTER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Performance Forecaster":
    st.markdown('<div class="main-title">Performance Forecaster</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Predict training time &amp; val loss for new hyperparameter configurations</div>',
        unsafe_allow_html=True
    )

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor

    if os.path.exists(RUN_LOGS):
        df_runs = pd.read_csv(RUN_LOGS)
        feature_cols = ["vocab_size", "embedding_dim", "context_block_size", "batch_size", "learning_rate"]
        X_runs = df_runs[feature_cols].values
        y_loss = df_runs["val_loss"].values
        y_time = df_runs["train_time_seconds"].values

        lin_reg  = LinearRegression().fit(X_runs, y_loss)
        poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X_runs, y_loss)
        knn_reg  = KNeighborsRegressor(n_neighbors=3).fit(X_runs, y_loss)
        dt_reg   = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_runs, y_loss)
        time_reg = LinearRegression().fit(X_runs, y_time)

        st.markdown("#### Try your own hyperparameters:")

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            vs  = st.number_input("Vocab Size",    min_value=10,     max_value=500,  value=100,   step=10)
        with col_b:
            ed  = st.number_input("Embedding Dim", min_value=8,      max_value=512,  value=64,    step=8)
        with col_c:
            cbs = st.number_input("Context Block", min_value=4,      max_value=128,  value=16,    step=4)
        with col_d:
            bs  = st.number_input("Batch Size",    min_value=8,      max_value=256,  value=64,    step=8)
        with col_e:
            lr_val = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")

        X_new = np.array([[vs, ed, cbs, bs, lr_val]])

        lin_pred  = lin_reg.predict(X_new)[0]
        poly_pred = poly_reg.predict(X_new)[0]
        knn_pred  = knn_reg.predict(X_new)[0]
        dt_pred   = dt_reg.predict(X_new)[0]
        time_pred = time_reg.predict(X_new)[0]

        st.markdown("---")
        st.markdown("##### Predicted Val Loss for your config:")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Linear Regression",    f"{lin_pred:.4f}")
        mc2.metric("Polynomial Regression", f"{poly_pred:.4f}")
        mc3.metric("KNN Regressor",         f"{knn_pred:.4f}")
        mc4.metric("Decision Tree",         f"{dt_pred:.4f}")

        st.metric(
            "Estimated Training Time",
            f"{max(time_pred, 1):.0f}s  (~{max(time_pred,1)/60:.1f} min)",
            help="Estimated from linear regression on run_logs.csv"
        )

        st.markdown("---")
        st.markdown("#### Training Runs Log")
        st.dataframe(
            df_runs.style.format({
                "val_loss": "{:.4f}",
                "learning_rate": "{:.4f}",
                "train_time_seconds": "{:.0f}s"
            }).background_gradient(subset=["val_loss"], cmap="RdYlGn_r"),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("#### Val Loss Forecast vs Future Training Steps")
        if os.path.exists(METRICS_PATH):
            df_m = pd.read_csv(METRICS_PATH)
            x_s  = df_m["step"].values.reshape(-1, 1)
            y_s  = df_m["val_loss"].values

            lin_s  = LinearRegression().fit(x_s, y_s)
            poly_s = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(x_s, y_s)
            dt_s   = DecisionTreeRegressor(max_depth=3, random_state=42).fit(x_s, y_s)

            x_plot = np.linspace(0, 100000, 400).reshape(-1, 1)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.scatter(x_s, y_s, s=6, alpha=0.35, color="gray", label="Actual Val Loss")
            ax3.plot(x_plot, lin_s.predict(x_plot),  color="#4f8ef7", label="Linear Regression")
            ax3.plot(x_plot, poly_s.predict(x_plot), color="#4fc97f", label="Polynomial Regression")
            ax3.plot(x_plot, dt_s.predict(x_plot),   color="#f76b4f", linestyle="--", label="Decision Tree Regressor")
            ax3.axvline(int(df_m["step"].iloc[-1]), color="gray", linestyle=":", linewidth=1.5, label="Current step")
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Val Loss")
            ax3.set_title("Val Loss Forecast Beyond Current Training")
            ax3.legend()
            ax3.grid(True, alpha=0.25)
            ax3.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "How It Works":
    st.markdown('<div class="main-title">How AutoLingo Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A plain-English explanation of every algorithm powering this dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("## The Core Idea: Character-Level Prediction")
    st.markdown("""
AutoLingo learns **which character comes next** by looking at the **3 most recent characters** (context window).

For example, given the text `"hel"`, each model tries to predict: *what character follows?* (Answer: `"l"`)

This window slides over all training data so the models learn millions of patterns.
    """)
    st.code('context = ["h", "e", "l"]   →   predict: "l"', language="text")

    st.markdown("---")

    with st.expander("**KNN Classifier (K-Nearest Neighbours)**", expanded=True):
        st.markdown("""
**How it works:**
1. Every 3-character context is converted into 3 numbers (character IDs).
2. For a new context, KNN finds the **K=5 closest** training contexts (by Euclidean distance).
3. It takes a **majority vote** among those 5 neighbours — the most common next character wins.
4. `predict_proba()` returns the fraction of neighbours that voted for each character.

**Strength:** Simple, no assumptions, works well when similar contexts have similar next characters.

**Weakness:** Slow at prediction for large datasets. Struggles with rare character combinations.

```
"hel"  →  [7, 4, 11]  →  5 nearest contexts  →  vote  →  "l"
```
        """)

    with st.expander("**Decision Tree Classifier**", expanded=False):
        st.markdown("""
**How it works:**
1. The model learns a tree of `if/else` rules from the training data.
2. At each node: *"Is Char1 ID < 30? If yes, go left. If no, go right."*
3. Leaf nodes store the most likely next character.
4. `predict_proba()` returns the class distribution in the leaf node.

**Strength:** Very fast prediction. Explainable — you can trace exactly why a prediction was made.

**Weakness:** Can overfit (memorize training data) if the tree is too deep.

```
if Char3 == 'l'  →  predict 'l'
else if Char2 == 'e'  →  predict 'o'
...
```
        """)

    with st.expander("**Label Propagation (Semi-Supervised)**", expanded=False):
        st.markdown("""
**How it works:**
Label Propagation is a **semi-supervised** algorithm — it can learn from both **labeled** and **unlabeled** data.

1. Build a similarity graph: each training context is a node. Nearby contexts are connected.
2. Start with a subset of known labels (30% labeled, 70% unknown).
3. Labels "flow" through the graph edges — similar contexts get similar labels.
4. After convergence, even unlabeled points have a predicted label.

**Strength:** Uses unlabeled data efficiently. Good at generalizing from small labeled sets.

**Weakness:** Slower to train. Graph-based — needs enough data to form good neighbourhoods.
        """)

    with st.expander("**Logistic Regression** *(NEW)*", expanded=False):
        st.markdown("""
**How it works:**
1. Fits a linear decision boundary in the 3-dimensional feature space.
2. Uses the **softmax** function to output a probability for *every* character class.
3. Trained with L-BFGS optimisation (max 500 iterations).

**Strength:** Produces **well-calibrated probabilities** — great for the Top-K view.
Trains fast, generalises well with regularisation (C=1.0).

**Weakness:** Linear — cannot capture non-linear character relationships without feature engineering.

$$P(y = k \\mid x) = \\frac{e^{W_k \\cdot x}}{\\sum_j e^{W_j \\cdot x}}$$
        """)

    with st.expander("**Random Forest Classifier** *(NEW)*", expanded=False):
        st.markdown("""
**How it works:**
1. Trains **100 decision trees**, each on a random subset of training samples (bagging).
2. Each tree also considers a random subset of features at every split.
3. Final prediction = **majority vote** across all 100 trees.
4. `predict_proba()` = average fraction of trees that voted for each character.

**Strength:** Robust against overfitting. Often outperforms single Decision Trees.
Parallel training (n_jobs=-1) makes it fast.

**Weakness:** Less interpretable than a single tree. Slightly more memory usage.
        """)

    st.markdown("---")
    st.markdown("## 💾 Model Persistence (joblib)")
    st.markdown("""
All 5 trained classifiers are **saved to disk** at `models/autolingo_models.joblib` using `joblib.dump()`.

On every subsequent dashboard launch, models are **loaded from disk** instead of being retrained:

```python
joblib.dump({"knn": knn, "dt": dt, ...}, "models/autolingo_models.joblib")
# later…
data = joblib.load("models/autolingo_models.joblib")
```

Use the **Retrain & overwrite cache** button in the sidebar to force a full retrain (e.g., after adding new training data).
    """)

    st.markdown("---")
    st.markdown("## Performance Forecaster")
    st.markdown("""
The forecaster answers: *"If I change hyperparameters, how will the model perform?"*

| Regressor | Method |
|---|---|
| **Linear Regression** | Fits a straight line: `val_loss = a*vocab_size + b*embedding_dim + ...` |
| **Polynomial Regression** | Same but adds squared/interaction terms for curved relationships |
| **KNN Regressor** | Finds the K most similar past training runs and averages their outcomes |
| **Decision Tree Regressor** | Learns rule-based thresholds for hyperparameters → outcome |

Trained on `run_logs.csv` — 6 real training runs with different configurations.
    """)

    st.markdown("---")

    st.markdown("## Benchmark Metrics")
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("""
**Accuracy** — What fraction of next-character predictions were exactly correct?

$$\\text{Accuracy} = \\frac{\\text{Correct Predictions}}{\\text{Total Predictions}}$$
        """)
    with col_y:
        st.markdown("""
**F1 Score (Macro)** — Balances precision and recall across all character classes.

$$F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$
        """)

    st.info("""
**Why are the accuracy numbers low (~20-40%)?**

Predicting the exact next character out of 50-100 possible characters is a very hard task.
Even humans can't always guess the next letter. The important thing is the *relative* comparison
between models, not the absolute value. Label Propagation leading the accuracy chart shows
semi-supervised learning has an edge with this dataset.
    """)

    st.markdown("---")
    st.markdown("### AutoLingo System Flow")
    st.code("""
[Raw Text] (Member 1)
     |
     v  Label Propagation filters clean sentences
[Clean Training Corpus]
     |
     +------------------------+
     v                        v
[KNN / DT / LabelProp /    [PyTorch Neural Net]  (Member 3 - Harsha)
 Logistic Reg / RF]              |
(Member 2 - Sibling B)           v
     |                   [Benchmark Compare] (Member 4 - YOU)
     +──────────────────────────>|
                                 v
                       [FastAPI + Streamlit Dashboard] (Member 5)
    """, language="text")
