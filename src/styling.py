"""Theme helpers: global CSS, Plotly template and small HTML components."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---- hero banner ---- */
.noon-hero {{
  background: linear-gradient(120deg, {C.NOON_YELLOW} 0%, #FFD500 55%, #F5C400 100%);
  border-radius: 18px; padding: 22px 30px; margin-bottom: 18px; color: #0B0B0C;
  display: flex; align-items: center; justify-content: space-between; gap: 18px;
  box-shadow: 0 10px 30px rgba(254,238,0,.12);
}}
.noon-hero .brand {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -1.5px; line-height: 1; }}
.noon-hero .brand span {{ background:#0B0B0C; color:{C.NOON_YELLOW}; border-radius: 10px;
  padding: 2px 12px 5px; margin-right: 12px; }}
.noon-hero .tag {{ font-size: .95rem; font-weight: 600; opacity: .85; text-align:right; }}

/* ---- KPI cards ---- */
.kpi {{ background: {C.PANEL}; border: 1px solid {C.GRID}; border-left: 4px solid {C.NOON_YELLOW};
  border-radius: 12px; padding: 14px 18px; height: 100%; }}
.kpi .lbl {{ color: {C.MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.kpi .val {{ font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin-top: 2px; }}
.kpi .sub {{ color: {C.MUTED}; font-size: .8rem; margin-top: 2px; }}
.kpi.red {{ border-left-color: {C.FRAUD_RED}; }} .kpi.green {{ border-left-color: {C.SAFE_GREEN}; }}
.kpi.cyan {{ border-left-color: {C.CYAN}; }}
.kpi .up {{ color: {C.SAFE_GREEN}; }} .kpi .down {{ color: {C.FRAUD_RED}; }}

/* ---- section headers ---- */
.sec {{ display:flex; align-items:center; gap:10px; margin: 6px 0 4px; }}
.sec .bar {{ width: 6px; height: 24px; background: {C.NOON_YELLOW}; border-radius: 3px; }}
.sec h3 {{ margin: 0; font-size: 1.3rem; font-weight: 700; }}
.sec-sub {{ color: {C.MUTED}; margin: 0 0 12px 16px; font-size: .92rem; }}

/* ---- 7-step pipeline strip ---- */
.steps {{ display:grid; grid-template-columns: repeat(7, 1fr); gap:8px; margin: 2px 0 14px; }}
.step {{ background:{C.PANEL}; border:1px solid {C.GRID}; border-top:3px solid {C.GRID}; border-radius:10px;
  padding:8px 10px; font-size:.78rem; line-height:1.25; }}
.step .n {{ color:{C.MUTED}; font-weight:800; font-size:.72rem; letter-spacing:.06em; }}
.step .t {{ font-weight:700; margin-top:2px; }}
.step .m {{ color:{C.MUTED}; margin-top:3px; font-size:.72rem; }}
.step.star {{ border-top-color:{C.NOON_YELLOW}; }}
.step.active {{ background:{C.PANEL_2}; border-color:{C.NOON_YELLOW}; box-shadow:0 0 0 1px {C.NOON_YELLOW} inset; }}
.step.active .n {{ color:{C.NOON_YELLOW}; }}
@media (max-width: 1000px) {{ .steps {{ grid-template-columns: repeat(2, 1fr); }} }}

/* ---- verdict pill ---- */
.verdict {{ border-radius: 14px; padding: 16px 22px; font-weight: 800; font-size: 1.5rem; text-align:center; }}
.verdict.fraud {{ background: rgba(255,90,95,.14); border: 2px solid {C.FRAUD_RED}; color: {C.FRAUD_RED}; }}
.verdict.legit {{ background: rgba(46,196,182,.12); border: 2px solid {C.SAFE_GREEN}; color: {C.SAFE_GREEN}; }}
.verdict small {{ display:block; font-size:.8rem; font-weight:500; color:{C.MUTED}; margin-top:4px; }}

/* ---- reasons list ---- */
.reason {{ background:{C.PANEL}; border:1px solid {C.GRID}; border-radius:10px; padding:8px 12px; margin-bottom:6px; font-size:.92rem; }}
.reason b.f {{ color:{C.FRAUD_RED}; }} .reason b.s {{ color:{C.SAFE_GREEN}; }}

/* ---- Streamlit widgets ---- */
div.stButton > button, div.stDownloadButton > button {{
  border-radius: 10px; font-weight: 700; border: 1px solid {C.GRID}; }}
div.stButton > button[kind="primary"] {{ background: {C.NOON_YELLOW}; color: #0B0B0C; border: none; }}
div.stButton > button[kind="primary"]:hover {{ background: #FFF85C; color: #0B0B0C; }}
div[data-testid="stExpander"] {{ border: 1px solid {C.GRID}; border-radius: 12px; background: {C.PANEL}; }}
div[data-testid="stExpander"] summary {{ font-weight: 600; }}
div[data-testid="stSidebar"] {{ border-right: 1px solid {C.GRID}; }}
div[data-testid="stSidebar"] .sb-brand {{ font-size: 1.7rem; font-weight: 800; letter-spacing:-1px; color:{C.NOON_YELLOW}; }}
div[data-testid="stSidebar"] .sb-sub {{ color:{C.MUTED}; font-size:.8rem; margin-bottom: 8px; }}
div[data-testid="stMetric"] {{ background: {C.PANEL}; border: 1px solid {C.GRID}; border-radius: 12px; padding: 10px 14px; }}
hr {{ border-color: {C.GRID}; }}
</style>
"""


def register_plotly_template() -> None:
    """Dark / yellow Plotly template applied to every figure."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C.TEXT, size=13),
        colorway=[C.NOON_YELLOW, C.CYAN, C.PINK, C.FRAUD_RED, C.SAFE_GREEN, C.ORANGE, C.PURPLE],
        xaxis=dict(gridcolor=C.GRID, zerolinecolor=C.GRID, linecolor=C.GRID),
        yaxis=dict(gridcolor=C.GRID, zerolinecolor=C.GRID, linecolor=C.GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=C.PANEL_2, font_color=C.TEXT),
        title=dict(font=dict(size=16)),
    )
    pio.templates["noon"] = tpl
    pio.templates.default = "noon"


def inject_css() -> None:
    register_plotly_template()
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, tagline: str) -> None:
    st.markdown(
        f'<div class="noon-hero"><div class="brand"><span>noon</span>{title}</div>'
        f'<div class="tag">{tagline}</div></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sec"><div class="bar"></div><h3>{title}</h3></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sec-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    """Return the HTML of a KPI card (tone: '', 'red', 'green', 'cyan')."""
    return (f'<div class="kpi {tone}"><div class="lbl">{label}</div><div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def show_kpi(col, *args, **kw) -> None:
    col.markdown(kpi(*args, **kw), unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, template="noon")
    if title:
        fig.update_layout(title=title)
    return fig


def plot(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
