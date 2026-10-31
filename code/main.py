
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="야구 타구 데이터 분석",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ 야구 타구 데이터 분석")
st.caption("타구 속도, 발사각, 비거리, 배럴 비율 분석")

DATA_URL = "https://raw.githubusercontent.com/leeseojin09/baseball/refs/heads/main/data/exit_velocity.csv"

@st.cache_data
def load_data():
    response = requests.get(DATA_URL, timeout=20)
    response.raise_for_status()

    from io import BytesIO
    data = pd.read_csv(BytesIO(response.content))

    # 선수 이름 열 확인
    if "last_name, first_name" in data.columns:
        data["last_name, first_name"] = (
            data["last_name, first_name"].fillna("이름 없음")
        )

    # 수치형 열 변환
    numeric_cols = [
        "avg_hit_angle",
        "avg_hit_speed",
        "avg_distance",
        "brl_percent"
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# ---------------------------------
# 원본 데이터 전체 보기
# ---------------------------------
st.header("📋 기존 데이터 전체 보기")

st.write(
    f"전체 데이터: {len(df):,}행 × {len(df.columns)}열"
)

with st.expander("원본 데이터 확인하기", expanded=True):
    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    st.download_button(
        "📥 원본 데이터 전체 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="exit_velocity_original.csv",
        mime="text/csv"
    )

st.divider()


# ---------------------------------
# 분석에 사용할 데이터
# ---------------------------------
st.header("📊 타구 데이터 분석")

numeric_cols = [
    "avg_hit_angle",
    "avg_hit_speed",
    "avg_distance",
    "brl_percent"
]

available_cols = [col for col in numeric_cols if col in df.columns]

if not available_cols:
    st.error("분석할 수치형 데이터가 없습니다.")
    st.stop()

analysis_df = df.dropna(subset=available_cols).copy()

# ---------------------------------
# 데이터 요약
# ---------------------------------
st.subheader("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 선수 데이터", f"{len(df):,}개")
col2.metric("분석 가능한 데이터", f"{len(analysis_df):,}개")
col3.metric("분석 항목", f"{len(available_cols)}개")

st.subheader("기초 통계량")
st.dataframe(
    analysis_df[available_cols].describe().round(2),
    use_container_width=True
)


# ---------------------------------
# 선수 이름 포함 분석 데이터
# ---------------------------------
name_col = "last_name, first_name"

if name_col in analysis_df.columns:
    display_cols = [name_col] + available_cols
else:
    display_cols = available_cols

st.subheader("2. 선수별 데이터")

st.dataframe(
    analysis_df[display_cols],
    use_container_width=True
)


# ---------------------------------
# 히스토그램
# ---------------------------------
st.subheader("3. 항목별 데이터 분포")

selected_hist = st.selectbox(
    "분포를 확인할 항목",
    available_cols
)

fig_hist = px.histogram(
    analysis_df,
    x=selected_hist,
    nbins=20,
    title=f"{selected_hist} 분포"
)

st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------
# 선수별 순위
# ---------------------------------
st.subheader("4. 선수별 순위")

rank_col = st.selectbox(
    "순위를 확인할 항목",
    available_cols,
    key="rank"
)

rank_df = analysis_df.sort_values(
    rank_col,
    ascending=False
).head(20)

if name_col in rank_df.columns:
    fig_rank = px.bar(
        rank_df,
        x=name_col,
        y=rank_col,
        title=f"{rank_col} 상위 20명",
        hover_data=available_cols
    )
else:
    fig_rank = px.bar(
        rank_df,
        y=rank_col,
        title=f"{rank_col} 상위 20명"
    )

st.plotly_chart(fig_rank, use_container_width=True)


# ---------------------------------
# 상관관계 분석
# ---------------------------------
st.subheader("5. 항목 간 상관관계")

corr = analysis_df[available_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="타구 데이터 상관관계"
)

st.plotly_chart(fig_corr, use_container_width=True)


# ---------------------------------
# 산점도 및 회귀분석
# ---------------------------------
st.subheader("6. 두 변수의 관계 분석")

col1, col2 = st.columns(2)

with col1:
    x_col = st.selectbox(
        "독립변수 (X)",
        available_cols,
        index=available_cols.index("avg_hit_speed")
        if "avg_hit_speed" in available_cols else 0
    )

with col2:
    y_col = st.selectbox(
        "종속변수 (Y)",
        available_cols,
        index=available_cols.index("avg_distance")
        if "avg_distance" in available_cols else min(1, len(available_cols)-1)
    )

scatter_df = analysis_df.dropna(subset=[x_col, y_col]).copy()

hover_name = name_col if name_col in scatter_df.columns else None

fig_scatter = px.scatter(
    scatter_df,
    x=x_col,
    y=y_col,
    hover_name=hover_name,
    title=f"{x_col}와 {y_col}의 관계"
)

# 직접 회귀선 계산
if len(scatter_df) >= 2 and scatter_df[x_col].nunique() >= 2:
    x = scatter_df[x_col].to_numpy()
    y = scatter_df[y_col].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    fig_scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="선형 회귀선"
        )
    )

    r = scatter_df[x_col].corr(scatter_df[y_col])
    r_squared = r ** 2

    st.metric("피어슨 상관계수", f"{r:.3f}")
    st.metric("결정계수 R²", f"{r_squared:.3f}")

    st.write(f"회귀식: y = {slope:.3f}x + {intercept:.3f}")

else:
    st.warning("회귀분석을 수행하기에 데이터가 부족합니다.")

st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------
# 분석 데이터 다운로드
# ---------------------------------
st.divider()
st.subheader("📥 분석 데이터 다운로드")

st.download_button(
    "분석 데이터 CSV 다운로드",
    data=analysis_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="exit_velocity_analysis.csv",
    mime="text/csv"
)

st.success("데이터 분석이 완료되었습니다!")
