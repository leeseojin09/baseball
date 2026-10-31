
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from io import BytesIO

# -----------------------------------
# 1. 페이지 설정
# -----------------------------------
st.set_page_config(
    page_title="야구 타구 데이터 기초 통계 분석실",
    page_icon="⚾",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "leeseojin09/baseball/refs/heads/main/"
    "data/exit_velocity.csv"
)

st.title("⚾ 야구 타구 데이터 기초 통계 분석실")
st.caption(
    "실제 CSV 데이터를 이용해 타구 속도, 발사각, 비거리, "
    "배럴 등 야구 타구 기록을 시각화하고 분석합니다."
)


# -----------------------------------
# 2. 데이터 불러오기
# -----------------------------------
@st.cache_data(ttl=3600)
def load_data():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(
        BytesIO(response.content),
        encoding="utf-8-sig"
    )

    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(axis=1, how="all")

    # 숫자로 변환할 수 있는 열만 숫자형으로 변환
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("오류 종류:", type(e).__name__)
    st.code(str(e))
    st.write("CSV 주소:")
    st.code(DATA_URL)
    st.info(
        "CSV 주소와 GitHub 파일 경로를 확인하고 "
        "requirements.txt에 필요한 라이브러리가 있는지 확인해 주세요."
    )
    st.stop()

if df.empty:
    st.error("CSV 파일은 읽었지만 데이터가 비어 있습니다.")
    st.stop()


# -----------------------------------
# 3. 데이터 기본 정보
# -----------------------------------
st.subheader("1. 데이터 개요")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터 수", f"{len(df):,}개")

with col2:
    st.metric("전체 변수 수", f"{len(df.columns):,}개")

with col3:
    st.metric("결측값 수", f"{int(df.isna().sum().sum()):,}개")

with st.expander("데이터 미리 보기", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("전체 열 이름"):
    st.write(list(df.columns))

with st.expander("열별 데이터 유형 및 결측값"):
    info_df = pd.DataFrame({
        "열 이름": df.columns,
        "데이터 유형": [str(df[col].dtype) for col in df.columns],
        "결측값 수": [int(df[col].isna().sum()) for col in df.columns],
        "결측 비율(%)": [
            round(df[col].isna().mean() * 100, 2)
            for col in df.columns
        ]
    })
    st.dataframe(info_df, use_container_width=True)


# -----------------------------------
# 4. 숫자형 변수 선택
# -----------------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not numeric_cols:
    st.error("분석할 숫자형 변수가 없습니다. CSV의 데이터 형식을 확인해 주세요.")
    st.stop()


# -----------------------------------
# 5. 기초 통계량
# -----------------------------------
st.subheader("2. 기초 통계량")

selected_stats = st.multiselect(
    "통계량을 확인할 변수 선택",
    options=numeric_cols,
    default=numeric_cols[:min(5, len(numeric_cols))]
)

if selected_stats:
    stats = df[selected_stats].describe().T
    stats = stats.rename(columns={
        "count": "개수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "50%": "중앙값",
        "75%": "3사분위수",
        "max": "최댓값"
    })
    st.dataframe(stats.round(3), use_container_width=True)
else:
    st.info("통계량을 확인할 변수를 선택해 주세요.")


# -----------------------------------
# 6. 히스토그램
# -----------------------------------
st.subheader("3. 데이터 분포 분석 — 히스토그램")

default_ev = numeric_cols.index("avg_hit_speed") if "avg_hit_speed" in numeric_cols else 0

hist_col = st.selectbox(
    "히스토그램으로 확인할 변수",
    options=numeric_cols,
    index=default_ev
)

hist_data = df[[hist_col]].dropna()

if not hist_data.empty:
    fig = px.histogram(
        hist_data,
        x=hist_col,
        nbins=20,
        title=f"{hist_col} 분포",
        labels={hist_col: hist_col, "count": "데이터 개수"},
        opacity=0.85
    )
    fig.update_layout(
        xaxis_title=hist_col,
        yaxis_title="데이터 개수",
        bargap=0.08
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"유효 데이터 수: {len(hist_data):,}개")
    st.write(f"평균: {hist_data[hist_col].mean():.3f}")
    st.write(f"중앙값: {hist_data[hist_col].median():.3f}")
else:
    st.warning("선택한 변수에 분석할 값이 없습니다.")


# -----------------------------------
# 7. 상자그림
# -----------------------------------
st.subheader("4. 이상값과 분포 분석 — 상자그림")

box_cols = st.multiselect(
    "상자그림으로 비교할 변수",
    options=numeric_cols,
    default=["avg_hit_speed"] if "avg_hit_speed" in numeric_cols else numeric_cols[:1]
)

if box_cols:
    box_df = df[box_cols].melt(
        var_name="변수",
        value_name="값"
    ).dropna()

    if not box_df.empty:
        fig = px.box(
            box_df,
            x="변수",
            y="값",
            color="변수",
            points="outliers",
            title="변수별 상자그림"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "상자그림의 점은 일반적인 기준에서 벗어난 이상값 후보입니다. "
            "반드시 오류 데이터라는 뜻은 아닙니다."
        )
    else:
        st.warning("선택한 변수에 분석할 값이 없습니다.")
else:
    st.info("비교할 변수를 하나 이상 선택해 주세요.")


# -----------------------------------
# 8. 선수별 기록 순위
# -----------------------------------
st.subheader("5. 선수별 기록 순위")

possible_name_cols = [
    "last_name, first_name",
    "player_name",
    "player",
    "name",
    "선수명",
    "선수 이름"
]

name_col = next((col for col in possible_name_cols if col in df.columns), None)

if name_col is None:
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    if text_cols:
        name_col = st.selectbox(
            "선수 이름이 들어 있는 열 선택",
            options=text_cols
        )

if name_col is not None:
    rank_col = st.selectbox(
        "순위를 확인할 기록",
        options=numeric_cols,
        index=default_ev
    )

    rank_order = st.radio(
        "순위 정렬 방식",
        options=["높은 순", "낮은 순"],
        horizontal=True
    )

    rank_count = st.slider(
        "표시할 선수 수",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    rank_df = df[[name_col, rank_col]].dropna().copy()

    rank_df = rank_df[
        rank_df[name_col].astype(str).str.strip() != ""
    ]

    rank_df = (
        rank_df.groupby(name_col, as_index=False)[rank_col]
        .mean()
        .sort_values(rank_col, ascending=(rank_order == "낮은 순"))
        .head(rank_count)
    )

    rank_df.insert(0, "순위", range(1, len(rank_df) + 1))

    st.dataframe(
        rank_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    if not rank_df.empty:
        chart_df = rank_df.sort_values(rank_col, ascending=True)

        fig = px.bar(
            chart_df,
            x=rank_col,
            y=name_col,
            orientation="h",
            title=f"선수별 {rank_col} 순위",
            labels={rank_col: rank_col, name_col: "선수"},
            hover_data={"순위": True}
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "선수 이름 열이 없어 선수별 순위 기능을 표시할 수 없습니다. "
        "다른 통계 분석 기능은 계속 사용할 수 있습니다."
    )


# -----------------------------------
# 9. 상관관계 분석
# -----------------------------------
st.subheader("6. 변수 간 상관관계 분석")

corr_cols = st.multiselect(
    "상관관계를 분석할 숫자형 변수 선택",
    options=numeric_cols,
    default=numeric_cols[:min(6, len(numeric_cols))]
)

if len(corr_cols) >= 2:
    corr_data = df[corr_cols].corr()

    st.write("상관계수 행렬")
    st.dataframe(corr_data.round(3), use_container_width=True)

    fig = px.imshow(
        corr_data,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="변수 간 상관관계 히트맵"
    )

    fig.update_layout(coloraxis_colorbar_title="상관계수")

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "상관계수는 -1부터 1 사이입니다. 절댓값이 1에 가까울수록 "
        "선형 관계가 강하고, 0에 가까울수록 선형 관계가 약합니다. "
        "상관관계만으로 인과관계를 판단할 수는 없습니다."
    )
else:
    st.info("상관관계를 분석하려면 변수를 2개 이상 선택해 주세요.")


# -----------------------------------
# 10. 산점도와 선형 추세선
# -----------------------------------

# -----------------------------------
# 10. 산점도와 선형 추세선
# -----------------------------------
st.subheader("7. 두 변수의 관계 — 산점도")

scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    x_col = st.selectbox(
        "X축 변수",
        options=numeric_cols,
        index=default_ev,
        key="scatter_x"
    )

with scatter_col2:
    y_col = st.selectbox(
        "Y축 변수",
        options=numeric_cols,
        index=1 if len(numeric_cols) > 1 else 0,
        key="scatter_y"
    )

scatter_df = df[[x_col, y_col]].dropna()

if len(scatter_df) >= 2:

    # 기본 산점도
    fig = px.scatter(
        scatter_df,
        x=x_col,
        y=y_col,
        title=f"{x_col}와 {y_col}의 관계",
        opacity=0.65
    )

    # 선형 회귀 추세선 직접 계산
    if x_col != y_col and scatter_df[x_col].nunique() >= 2:

        x = scatter_df[x_col].to_numpy(dtype=float)
        y = scatter_df[y_col].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        # Plotly에 직접 추세선 추가
        fig.add_scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="선형 회귀 추세선",
            line=dict(dash="dash", width=3)
        )

    st.plotly_chart(fig, use_container_width=True)

    if x_col != y_col:

        corr_value = scatter_df[x_col].corr(scatter_df[y_col])

        st.metric(
            "피어슨 상관계수",
            f"{corr_value:.4f}"
        )

        st.metric(
            "분석 데이터 수",
            f"{len(scatter_df):,}개"
        )

        if scatter_df[x_col].nunique() >= 2:
            r_squared = corr_value ** 2

            st.write("### 선형 회귀 분석 결과")
            st.write(
                f"회귀식: y = {slope:.4f}x + {intercept:.4f}"
            )
            st.write(f"결정계수(R²): {r_squared:.4f}")

else:
    st.warning("산점도를 그리기 위한 데이터가 부족합니다.")
