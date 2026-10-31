
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# =====================================
# 페이지 설정
# =====================================
st.set_page_config(
    page_title="야구 타구 데이터 분석",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ 야구 타구 데이터 분석")
st.caption("선수별 타구 속도, 발사각, 비거리, 배럴 비율 및 상관관계 분석")

DATA_URL = "https://raw.githubusercontent.com/leeseojin09/baseball/refs/heads/main/data/exit_velocity.csv"

NAME_COL = "last_name, first_name"

MAIN_COLS = [
    "avg_hit_angle",
    "avg_hit_speed",
    "avg_distance",
    "brl_percent"
]


# =====================================
# 데이터 불러오기
# =====================================
@st.cache_data
def load_data():
    response = requests.get(
        DATA_URL,
        timeout=30
    )
    response.raise_for_status()

    data = pd.read_csv(
        BytesIO(response.content)
    )

    # 열 이름 공백 제거
    data.columns = data.columns.astype(str).str.strip()

    # 선수 이름 처리
    if NAME_COL in data.columns:
        data[NAME_COL] = data[NAME_COL].fillna("이름 없음")

    return data


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =====================================
# 원본 데이터 전체 보기
# =====================================
st.header("📋 1. 원본 데이터 전체 보기")

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


# =====================================
# 분석용 숫자 데이터 생성
# =====================================
numeric_df = df.copy()

for col in numeric_df.columns:

    if col == NAME_COL:
        continue

    converted = pd.to_numeric(
        numeric_df[col],
        errors="coerce"
    )

    # 실제 숫자형 값이 존재하는 열만 숫자로 변환
    if converted.notna().any():
        numeric_df[col] = converted


# =====================================
# 기본 정보
# =====================================
st.header("📊 2. 데이터 기본 정보")

available_cols = [
    col for col in MAIN_COLS
    if col in numeric_df.columns
]

if not available_cols:
    st.error(
        "분석에 필요한 열을 찾을 수 없습니다. "
        "CSV의 실제 열 이름을 확인해 주세요."
    )

    st.write("현재 CSV 열 이름:")
    st.write(list(df.columns))
    st.stop()


analysis_df = numeric_df.copy()

for col in available_cols:
    analysis_df[col] = pd.to_numeric(
        analysis_df[col],
        errors="coerce"
    )

valid_df = analysis_df.dropna(
    subset=available_cols
).copy()

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 데이터",
    f"{len(df):,}개"
)

col2.metric(
    "분석 가능한 데이터",
    f"{len(valid_df):,}개"
)

col3.metric(
    "전체 열 개수",
    f"{len(df.columns)}개"
)

st.subheader("기초 통계량")

st.dataframe(
    valid_df[available_cols].describe().round(2),
    use_container_width=True
)


# =====================================
# 선수별 데이터
# =====================================
st.header("👤 3. 선수별 데이터")

if NAME_COL in valid_df.columns:

    display_cols = [
        NAME_COL
    ] + available_cols

    st.dataframe(
        valid_df[display_cols],
        use_container_width=True
    )

else:
    st.warning(
        "선수 이름 열을 찾을 수 없습니다."
    )


# =====================================
# 항목별 분포
# =====================================
st.header("📈 4. 항목별 데이터 분포")

selected_hist = st.selectbox(
    "분포를 확인할 항목",
    available_cols
)

fig_hist = px.histogram(
    valid_df,
    x=selected_hist,
    nbins=20,
    title=f"{selected_hist} 분포"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# =====================================
# 선수별 순위
# =====================================
st.header("🏆 5. 선수별 순위")

rank_col = st.selectbox(
    "순위를 확인할 항목",
    available_cols,
    key="rank"
)

rank_df = (
    valid_df
    .dropna(subset=[rank_col])
    .sort_values(rank_col, ascending=False)
    .head(20)
)

if NAME_COL in rank_df.columns:

    fig_rank = px.bar(
        rank_df,
        x=NAME_COL,
        y=rank_col,
        title=f"{rank_col} 상위 20개 데이터",
        hover_data=available_cols
    )

else:

    fig_rank = px.bar(
        rank_df,
        y=rank_col,
        title=f"{rank_col} 상위 20개 데이터"
    )

st.plotly_chart(
    fig_rank,
    use_container_width=True
)

st.dataframe(
    rank_df[
        [NAME_COL, rank_col]
    ] if NAME_COL in rank_df.columns
    else rank_df[[rank_col]],
    use_container_width=True
)


# =====================================
# 원본 데이터 전체 상관관계 분석
# =====================================
st.header("🔗 6. 원본 데이터 전체 상관관계 분석")

# 숫자형 열 자동 추출
original_numeric_df = numeric_df.select_dtypes(
    include=["number"]
).copy()

# 무한대 값을 결측치로 변경
original_numeric_df = original_numeric_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# 모든 값이 결측치인 열 제거
original_numeric_df = original_numeric_df.dropna(
    axis=1,
    how="all"
)

# 값이 하나뿐인 열 제거
original_numeric_df = original_numeric_df.loc[
    :,
    original_numeric_df.nunique(dropna=True) > 1
]

if original_numeric_df.shape[1] >= 2:

    st.write(
        f"분석 대상 수치형 항목: "
        f"{len(original_numeric_df.columns)}개"
    )

    # 피어슨 상관계수
    original_corr = original_numeric_df.corr(
        method="pearson"
    )

    st.subheader("전체 상관관계 히트맵")

    fig_original_corr = px.imshow(
        original_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="원본 데이터 전체 피어슨 상관관계"
    )

    st.plotly_chart(
        fig_original_corr,
        use_container_width=True
    )

    st.subheader("전체 상관관계 계수표")

    st.dataframe(
        original_corr.round(3),
        use_container_width=True
    )

    # 가장 강한 상관관계 계산
    corr_pairs = (
        original_corr.abs()
        .where(
            np.triu(
                np.ones(original_corr.shape),
                k=1
            ).astype(bool)
        )
        .stack()
        .sort_values(ascending=False)
    )

    st.subheader("상관관계가 강한 항목 조합")

    if not corr_pairs.empty:

        result_rows = []

        for (col1, col2), abs_value in corr_pairs.items():

            result_rows.append({
                "항목 1": col1,
                "항목 2": col2,
                "상관계수": original_corr.loc[col1, col2],
                "상관계수 절댓값": abs_value
            })

        result_df = pd.DataFrame(result_rows)

        st.dataframe(
            result_df.round(3),
            use_container_width=True
        )

        strongest = result_df.iloc[0]

        st.info(
            f"가장 강한 상관관계는 "
            f"'{strongest['항목 1']}'과 "
            f"'{strongest['항목 2']}' 사이에서 나타났으며, "
            f"상관계수는 {strongest['상관계수']:.3f}입니다."
        )

else:

    st.warning(
        "상관관계를 계산할 수 있는 수치형 항목이 "
        "2개 미만입니다."
    )


# =====================================
# 주요 타구 변수 상관관계
# =====================================
st.header("⚾ 7. 주요 타구 변수 상관관계")

main_numeric_df = valid_df[
    available_cols
].dropna()

if len(available_cols) >= 2:

    main_corr = main_numeric_df.corr(
        method="pearson"
    )

    fig_main_corr = px.imshow(
        main_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="주요 타구 변수 상관관계"
    )

    st.plotly_chart(
        fig_main_corr,
        use_container_width=True
    )

else:

    st.info(
        "주요 타구 변수 중 상관관계를 계산할 항목이 부족합니다."
    )


# =====================================
# 산점도 및 회귀분석
# =====================================
st.header("📉 8. 두 변수의 관계 및 회귀분석")

col1, col2 = st.columns(2)

with col1:

    x_default = (
        available_cols.index("avg_hit_speed")
        if "avg_hit_speed" in available_cols
        else 0
    )

    x_col = st.selectbox(
        "독립변수 (X)",
        available_cols,
        index=x_default
    )

with col2:

    y_default = (
        available_cols.index("avg_distance")
        if "avg_distance" in available_cols
        else min(1, len(available_cols) - 1)
    )

    y_col = st.selectbox(
        "종속변수 (Y)",
        available_cols,
        index=y_default
    )


scatter_df = valid_df.dropna(
    subset=[x_col, y_col]
).copy()

fig_scatter = px.scatter(
    scatter_df,
    x=x_col,
    y=y_col,
    hover_name=(
        NAME_COL
        if NAME_COL in scatter_df.columns
        else None
    ),
    title=f"{x_col}와 {y_col}의 관계"
)


if (
    len(scatter_df) >= 2
    and scatter_df[x_col].nunique() >= 2
):

    x = scatter_df[x_col].to_numpy()
    y = scatter_df[y_col].to_numpy()

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

    fig_scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="선형 회귀선"
        )
    )

    r = scatter_df[x_col].corr(
        scatter_df[y_col]
    )

    r_squared = r ** 2

    col1, col2 = st.columns(2)

    col1.metric(
        "피어슨 상관계수",
        f"{r:.3f}"
    )

    col2.metric(
        "결정계수 R²",
        f"{r_squared:.3f}"
    )

    st.write(
        f"회귀식: y = {slope:.3f}x + {intercept:.3f}"
    )

else:

    st.warning(
        "회귀분석을 수행하기에 데이터가 부족합니다."
    )


st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# =====================================
# 분석 데이터 다운로드
# =====================================
st.divider()

st.header("📥 9. 분석 결과 다운로드")

st.download_button(
    "분석 데이터 CSV 다운로드",
    data=valid_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="exit_velocity_analysis.csv",
    mime="text/csv"
)

st.success(
    "모든 데이터 분석이 완료되었습니다!"
)
