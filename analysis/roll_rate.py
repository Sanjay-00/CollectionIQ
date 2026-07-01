"""
Bucket Migration / Roll-Rate Analysis
Pure pandas - no LLM dependency.
Requires both current and previous month DataFrames matched on Loan No.
"""
import pandas as pd
import plotly.graph_objects as go

from utils import BUCKET_ORDER, BUCKET_SCORE

# Exclude NA bucket from migration analysis  -  not a meaningful delinquency state
VALID_BUCKETS = [b for b in BUCKET_ORDER if b != "NA"]

BUCKET_COLORS = {
    "STD":      "#16a34a",
    "1-30 DPD": "#FFC000",
    "SMA-1":    "#f97316",
    "SMA-2":    "#ef4444",
    "NPA":      "#991b1b",
}


def compute_roll_rate_matrix(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    key_col: str = "Loan No",
    bucket_col: str = "curr_bucket",
) -> tuple[pd.DataFrame, dict]:
    """
    Compute month-over-month bucket migration matrix.

    Returns (matrix_df, meta) where:
    - matrix_df: DataFrame[VALID_BUCKETS x VALID_BUCKETS], values = account counts
      rows = previous month bucket, columns = current month bucket
    - meta: dict with matched_count, new_entries, exits, plus the 3 headline rates
    """
    if len(df_prev) == 0 or key_col not in df_curr.columns or key_col not in df_prev.columns:
        empty = pd.DataFrame(0, index=VALID_BUCKETS, columns=VALID_BUCKETS)
        return empty, {"matched_count": 0, "new_entries": 0, "exits": 0,
                       "roll_forward_rate": 0.0, "roll_backward_rate": 0.0, "npa_formation_rate": 0.0}

    prev_keys = set(df_prev[key_col].dropna())
    curr_keys = set(df_curr[key_col].dropna())

    new_entries = len(curr_keys - prev_keys)
    exits       = len(prev_keys - curr_keys)

    prev_slim = df_prev[[key_col, bucket_col]].rename(columns={bucket_col: "prev_bucket"})
    curr_slim = df_curr[[key_col, bucket_col]].rename(columns={bucket_col: "curr_bucket"})

    merged = prev_slim.merge(curr_slim, on=key_col, how="inner")
    # Keep only accounts whose buckets are in the valid set
    merged = merged[merged["prev_bucket"].isin(VALID_BUCKETS) & merged["curr_bucket"].isin(VALID_BUCKETS)]

    if len(merged) == 0:
        empty = pd.DataFrame(0, index=VALID_BUCKETS, columns=VALID_BUCKETS)
        return empty, {"matched_count": 0, "new_entries": new_entries, "exits": exits,
                       "roll_forward_rate": 0.0, "roll_backward_rate": 0.0, "npa_formation_rate": 0.0}

    matrix = pd.crosstab(
        merged["prev_bucket"],
        merged["curr_bucket"],
    ).reindex(index=VALID_BUCKETS, columns=VALID_BUCKETS, fill_value=0)

    kpis = compute_roll_rate_kpis(matrix)
    meta = {
        "matched_count": len(merged),
        "new_entries":   new_entries,
        "exits":         exits,
        **kpis,
    }
    return matrix, meta


def compute_roll_rate_kpis(matrix: pd.DataFrame) -> dict:
    """Derive the 3 headline migration KPIs from the matrix.

    Roll forward: any account that moved to a strictly worse bucket (including STD → 1-30 DPD).
    Cure rate:    any account that moved to a strictly better bucket (including NPA → SMA-2).
    Both use all matched accounts as the denominator.
    NPA formation rate: non-NPA accounts that became NPA this month.
    """
    available_rows = [b for b in VALID_BUCKETS if b in matrix.index]
    total_matched = matrix.values.sum()

    rolled_forward = 0
    cured = 0
    for row in available_rows:
        prev_score = BUCKET_SCORE.get(row, -1)
        for col in matrix.columns:
            curr_score = BUCKET_SCORE.get(col, -1)
            count = matrix.loc[row, col]
            if curr_score > prev_score:
                rolled_forward += count
            elif curr_score < prev_score:
                cured += count

    roll_forward_rate = round(rolled_forward / total_matched * 100, 2) if total_matched > 0 else 0.0
    roll_backward_rate = round(cured          / total_matched * 100, 2) if total_matched > 0 else 0.0

    # NPA formation rate: non-NPA accounts (any bucket) that became NPA
    non_npa_rows = [b for b in available_rows if b != "NPA"]
    total_pre_npa = matrix.loc[non_npa_rows].values.sum() if non_npa_rows else 0
    formed_npa    = matrix.loc[non_npa_rows, "NPA"].sum() if non_npa_rows and "NPA" in matrix.columns else 0
    npa_formation_rate = round(formed_npa / total_pre_npa * 100, 2) if total_pre_npa > 0 else 0.0

    return {
        "roll_forward_rate":  roll_forward_rate,
        "roll_backward_rate":  roll_backward_rate,
        "npa_formation_rate": npa_formation_rate,
    }


def build_roll_rate_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """
    Annotated Plotly heatmap of the migration matrix.
    Diagonal = stable (neutral). Above diagonal = deterioration (red). Below = improvement (green).
    """
    buckets = [b for b in VALID_BUCKETS if b in matrix.index and b in matrix.columns]
    z_values = matrix.loc[buckets, buckets].values.tolist()
    total = sum(sum(row) for row in z_values) or 1

    # Build a directional color matrix: positive = worsened (red), negative = improved (green)
    color_matrix = []
    text_matrix  = []
    for i, row_bucket in enumerate(buckets):
        row_color = []
        row_text  = []
        for j, col_bucket in enumerate(buckets):
            count = z_values[i][j]
            prev_score = BUCKET_SCORE.get(row_bucket, 0)
            curr_score = BUCKET_SCORE.get(col_bucket, 0)
            direction = curr_score - prev_score   # positive = worsened, negative = improved
            row_color.append(direction * (count / total * 100))
            row_text.append(str(int(count)))
        color_matrix.append(row_color)
        text_matrix.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=color_matrix,
        x=[f"→ {b}" for b in buckets],
        y=[f"{b} →" for b in buckets],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#000"},
        # zmid=0 anchors "no change" (direction == 0) to the white midpoint of the
        # colorscale, so every diagonal / same-bucket cell (e.g. SMA-1 → SMA-1)
        # renders white regardless of how lopsided the improvements vs worsenings are.
        zmid=0,
        colorscale=[
            [0.0,  "#dcfce7"],   # strong green (improved)
            [0.45, "#f9fafb"],   # near white (stable)
            [0.5,  "#f9fafb"],   # diagonal / no change
            [0.55, "#fef2f2"],   # near white (mild worsening)
            [1.0,  "#991b1b"],   # dark red (severe worsening)
        ],
        showscale=False,
        hovertemplate="From %{y}<br>To %{x}<br>Accounts: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Bucket Migration Matrix (Prev Month → Curr Month)", font=dict(size=14, color="#000")),
        xaxis=dict(title="Current Month Bucket", tickfont=dict(color="#000"), side="bottom"),
        yaxis=dict(title="Previous Month Bucket", tickfont=dict(color="#000"), autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=45, b=20),
        height=340,
    )
    return fig
