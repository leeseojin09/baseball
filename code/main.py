
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
    page_title="야구 타구 데이터 분석실",
    page_icon="⚾",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "leeseojin09/baseball/refs/heads/main/"
    "data/exit_velocity.csv"
)

# 사용할 열 5개
TARGET_COLS = [
    "last_name, first_name",
    "avg_hit_angle",
    "avg_hit_speed",
    "avg_distance",
    "brl_percent"
]

# 숫자형 변수
NUMERIC_COLS = [
    "avg_hit_angle",
    "avg_hit_speed",
    "avg_distance",
    "brl_percent"
]

# 한글 변수명
KOREAN_NAMES = {
    "last_name, first_name": "선수 이름",
    "avg_hit_angle": "평균 발사각",
    "avg_hit_speed": "평균 타구 속도",
    "avg_distance": "평균 비거리",
    "brl_percent": "배럴 비율"
}


# -----------------------------------
# 2. 제목
# -----------------------------------
st.title("⚾ 야구 타구 데이터 분석실")

st.caption(
    "선수 이름과 4가지 타구 기록을 이용한 "
    "기초 통계 및 데이터 시각화 분석"
)


# -----------------------------------
# 3. 데이터 불러오기
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

    # 필요한 열 확인
    missing_cols = [
        col for col in TARGET_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {missing_cols}"
        )

    # 지정한 5개 열만 남기기
    df = df[TARGET_COLS].copy()

    # 선수 이름 정리
    df["last_name, first_name"] = (
        df["last_name, first_name"]
        .fillna("이름 없음")
        .astype(str)
        .str.strip()
    )

    # 숫자형 변수 변환
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("오류 종류:", type(e).__name__)
    st.code(str(e))
    st.stop()


# -----------------------------------
# 4. 데이터 개요
# -----------------------------------
st.subheader("1. 데이터 개요")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 선수 데이터", f"{len(df):,}개")

with col2:
    st.metric("분석 변수 수", "5개")

with col3:
    st.metric(
        "결측값 수",
        f"{int(df.isna().sum().sum()):,}개"
    )

st.write("### 분석 데이터 미리 보기")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.write("### 최종 분석 변수")

for col in TARGET_COLS:
    st.write(f"- {col} : {KOREAN_NAMES[col]}")


# -----------------------------------
# 5. 기초 통계량
# -----------------------------------
st.subheader("2. 기초 통계량")

stats = df[NUMERIC_COLS].describe().T

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

stats.index = [
    KOREAN_NAMES[col]
    for col in stats.index
]

st.dataframe(
    stats.round(3),
    use_container_width=True
)


# -----------------------------------
# 6. 히스토그램
# -----------------------------------
st.subheader("3. 변수별 데이터 분포")

hist_col = st.selectbox(
    "분포를 확인할 변수 선택",
    options=NUMERIC_COLS,
    format_func=lambda x: KOREAN_NAMES[x]
)

hist_data = df[[hist_col]].dropna()

if not hist_data.empty:

    fig = px.histogram(
        hist_data,
        x=hist_col,
        nbins=20,
        title=f"{KOREAN_NAMES[hist_col]} 분포",
        labels={
            hist_col: KOREAN_NAMES[hist_col],
            "count": "데이터 개수"
        },
        opacity=0.8
    )

    fig.update_layout(
        xaxis_title=KOREAN_NAMES[hist_col],
        yaxis_title="데이터 개수",
        bargap=0.08
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        f"평균: {hist_data[hist_col].mean():.3f}"
    )

    st.write(
        f"중앙값: {hist_data[hist_col].median():.3f}"
    )

else:
    st.warning("분석할 데이터가 없습니다.")


# -----------------------------------
# 7. 상자그림
# -----------------------------------
st.subheader("4. 변수별 이상값 분석")

box_df = df.melt(
    id_vars=["last_name, first_name"],
    value_vars=NUMERIC_COLS,
    var_name="변수",
    value_name="값"
).dropna()

box_df["변수명"] = box_df["변수"].map(
    KOREAN_NAMES
)

fig = px.box(
    box_df,
    x="변수명",
    y="값",
    color="변수명",
    points="outliers",
    hover_data=["last_name, first_name"],
    title="4개 변수의 상자그림 비교"
)

fig.update_layout(
    showlegend=False,
    xaxis_title="분석 변수",
    yaxis_title="측정값"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "상자그림의 점은 일반적인 기준에서 벗어난 이상값 후보입니다."
)


# -----------------------------------
# 8. 선수별 기록 순위
# -----------------------------------
st.subheader("5. 선수별 기록 순위")

rank_col = st.selectbox(
    "순위를 확인할 기록",
    options=NUMERIC_COLS,
    format_func=lambda x: KOREAN_NAMES[x]
)

rank_order = st.radio(
    "순위 정렬 방식",
    ["높은 순", "낮은 순"],
    horizontal=True
)

rank_count = st.slider(
    "표시할 선수 수",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

rank_df = df[
    ["last_name, first_name", rank_col]
].dropna().copy()

rank_df = rank_df.sort_values(
    rank_col,
    ascending=(rank_order == "낮은 순")
).head(rank_count)

rank_df = rank_df.rename(columns={
    "last_name, first_name": "선수 이름",
    rank_col: KOREAN_NAMES[rank_col]
})

rank_df.insert(
    0,
    "순위",
    range(1, len(rank_df) + 1)
)

st.dataframe(
    rank_df.round(3),
    use_container_width=True,
    hide_index=True
)

# 선수별 막대그래프
if not rank_df.empty:

    fig = px.bar(
        rank_df.sort_values(
            KOREAN_NAMES[rank_col],
            ascending=True
        ),
        x=KOREAN_NAMES[rank_col],
        y="선수 이름",
        orientation="h",
        title=f"선수별 {KOREAN_NAMES[rank_col]} 순위"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -----------------------------------
# 9. 상관관계 분석
# -----------------------------------
st.subheader("6. 변수 간 상관관계 분석")

corr_data = df[NUMERIC_COLS].corr()

corr_display = corr_data.copy()

corr_display.index = [
    KOREAN_NAMES[col]
    for col in corr_display.index
]

corr_display.columns = [
    KOREAN_NAMES[col]
    for col in corr_display.columns
]

st.write("### 상관계수 행렬")

st.dataframe(
    corr_display.round(3),
    use_container_width=True
)

fig = px.imshow(
    corr_display,
    text_auto=".2f",
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="변수 간 상관관계 히트맵"
)

fig.update_layout(
    coloraxis_colorbar_title="상관계수"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "상관계수는 -1부터 1 사이입니다. "
    "절댓값이 클수록 선형 관계가 강합니다. "
    "상관관계만으로 인과관계를 판단할 수는 없습니다."
)


# -----------------------------------
# 10. 산점도와 선형 회귀 분석
# -----------------------------------
st.subheader("7. 두 변수의 관계 분석")

col1, col2 = st.columns(2)

with col1:
    x_col = st.selectbox(
        "X축 변수 선택",
        options=NUMERIC_COLS,
        index=1,
        format_func=lambda x: KOREAN_NAMES[x],
        key="x_axis"
    )

with col2:
    y_col = st.selectbox(
        "Y축 변수 선택",
        options=NUMERIC_COLS,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x],
        key="y_axis"
    )

scatter_df = df[
    ["last_name, first_name", x_col, y_col]
].dropna()

if len(scatter_df) >= 2:

    fig = px.scatter(
        scatter_df,
        x=x_col,
        y=y_col,
        hover_name="last_name, first_name",
        title=(
            f"{KOREAN_NAMES[x_col]}와 "
            f"{KOREAN_NAMES[y_col]}의 관계"
        ),
        opacity=0.7,
        labels={
            x_col: KOREAN_NAMES[x_col],
            y_col: KOREAN_NAMES[y_col]
        }
    )

    # 선형 회귀 추세선 직접 계산
    if (
        x_col != y_col
        and scatter_df[x_col].nunique() >= 2
    ):

        x = scatter_df[x_col].to_numpy(dtype=float)
        y = scatter_df[y_col].to_numpy(dtype=float)

        slope, intercept = np.polyfit(
            x,
            y,
            1
        )

        x_line = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_line = slope * x_line + intercept

        fig.add_scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="선형 회귀 추세선",
            line=dict(
                dash="dash",
                width=3
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if x_col != y_col:

        corr_value = scatter_df[x_col].corr(
            scatter_df[y_col]
        )

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "피어슨 상관계수",
                f"{corr_value:.4f}"
                if pd.notna(corr_value)
                else "계산 불가"
            )

        with metric2:
            st.metric(
                "분석 데이터 수",
                f"{len(scatter_df):,}개"
            )

        if (
            scatter_df[x_col].nunique() >= 2
            and pd.notna(corr_value)
        ):

            r_squared = corr_value ** 2

            st.write("### 선형 회귀 분석 결과")

            st.write(
                f"회귀식: y = {slope:.4f}x + {intercept:.4f}"
            )

            st.write(
                f"결정계수(R²): {r_squared:.4f}"
            )

else:
    st.warning(
        "산점도를 그리기 위한 데이터가 부족합니다."
    )


# -----------------------------------
# 11. 데이터 다운로드
# -----------------------------------
st.subheader("8. 분석 데이터 다운로드")

csv_data = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="📥 5개 열만 포함된 CSV 다운로드",
    data=csv_data,
    file_name="baseball_selected_data.csv",
    mime="text/csv"
)

st.success(
    "선수 이름과 4개 타구 변수를 이용한 분석이 완료되었습니다."
)

st.caption(
    "Plotly 그래프는 확대, 축소, 마우스 오버를 통한 "
    "수치 확인이 가능합니다."
)
