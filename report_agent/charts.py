"""Render Plotly figures to base64 PNG data URIs for embedding in the email-safe HTML report."""
import base64


def fig_to_base64(fig, width: int = 1000, height: int = 400, scale: float = 2.0) -> str | None:
    """Returns a data:image/png;base64,... URI, or None on any failure (missing kaleido, empty figure)."""
    try:
        if fig is None or not getattr(fig, "data", None):
            return None
        png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None
