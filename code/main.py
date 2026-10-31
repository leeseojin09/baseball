
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="야구 타구 데이터 분석실",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ 야구 타구 데이터 분석실")
st.caption("메이저리그 타구 데이터를 활용한 선수별 타격 능력 분석")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "leeseojin09/baseball/refs/heads/main/data/exit_velocity.csv"
)

NAME_COL = "last_name, first_name"

# 새롭게 변경한 주요 분석 항목 4개
MAIN_COLS = [
    "brl_percent",
    "fbld",
    "ev95percent",
    "avg_distance"
]

# --------------------------------------------------
# 2. 열 이름 한글 번역
# --------------------------------------------------

KOREAN_NAMES = {
    "last_name, first_name": "선수 이름",
    "player_id": "선수 ID",
    "attempts": "타구 데이터 수",
    "avg_hit_angle": "평균 발사각",
    "anglesweetspotpercent": "이상적인 발사각 비율",
    "max_hit_speed": "최고 타구 속도",
    "avg_hit_speed": "평균 타구 속도",
    "ev50": "타구 속도 중앙값",
    "fbld": "플라이볼·라인드라이브 비율",
    "gb": "땅볼 비율",
    "max_distance": "최대 타구 비거리",
    "avg_distance": "평균 타구 비거리",
    "avg_hr_distance": "평균 홈런 비거리",
    "ev95plus": "95마일 이상 타구 수",
    "ev95percent": "95마일 이상 타구 비율",
    "barrels": "배럴 타구 수",
    "brl_percent": "배럴 타구 비율",
    "brl_pa": "타석 대비 배럴 비율"
}


def korean_name(column):
    return KOREAN_NAMES.get(column, column)


# --------------------------------------------------
# 3. 데이터 불러오기
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    response = requests.get(DATA_URL, timeout=20)
    response.raise_for_status()

    df = pd.read_csv(BytesIO(response.content))

    df.columns = df.columns.str.strip()

    # 숫자형 데이터 변환
    for col in df.columns:
        if col != NAME_COL:
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().any():
                df[col] = converted

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 4. 데이터 확인
# --------------------------------------------------

if NAME_COL not in df.columns:
    st.error("선수 이름 열을 찾을 수 없습니다.")
    st.stop()

missing_cols = [col for col in MAIN_COLS if col not in df.columns]

if missing_cols:
    st.error(f"분석에 필요한 열이 없습니다: {missing_cols}")
    st.stop()

numeric_df = df.select_dtypes(include="number").copy()
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

st.success(
    f"데이터 불러오기 완료! "
    f"총 {len(df):,}명의 데이터와 {len(df.columns)}개 열을 확인했습니다."
)


# --------------------------------------------------
# 5. 전체 데이터 표시
# --------------------------------------------------

st.header("1. 전체 데이터")

st.write(
    "원본 CSV에 포함된 모든 선수의 데이터를 확인할 수 있습니다."
)

display_df = df.rename(columns=KOREAN_NAMES)

st.dataframe(
    display_df,
    use_container_width=True,
    height=400
)

st.download_button(
    label="전체 데이터 CSV 다운로드",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="baseball_data.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 6. 주요 분석 항목 4개
# --------------------------------------------------

st.header("2. 주요 타격 지표 분석")

st.write(
    "배럴 타구 비율, FB/LD 비율, "
    "95마일 이상 타구 비율, 평균 타구 비거리를 분석합니다."
)

metric_cols = st.columns(4)

for i, col in enumerate(MAIN_COLS):
    values = pd.to_numeric(df[col], errors="coerce").dropna()

    if len(values) > 0:
        metric_cols[i].metric(
            label=korean_name(col),
            value=f"{values.mean():.2f}"
        )
    else:
        metric_cols[i].metric(
            label=korean_name(col),
            value="데이터 없음"
        )


# --------------------------------------------------
# 7. 주요 항목 통계 요약
# --------------------------------------------------

st.subheader("주요 지표 통계 요약")

summary = numeric_df[MAIN_COLS].describe().T

summary.index = [
    korean_name(col) for col in summary.index
]

summary = summary.rename(
    columns={
        "count": "데이터 수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "50%": "중앙값",
        "75%": "3사분위수",
        "max": "최댓값"
    }
)

st.dataframe(
    summary.round(2),
    use_container_width=True
)


# --------------------------------------------------
# 8. 항목별 분포 그래프
# --------------------------------------------------

st.header("3. 주요 지표 분포 분석")

selected_metric = st.selectbox(
    "분포를 확인할 항목을 선택하세요.",
    MAIN_COLS,
    format_func=korean_name
)

hist_data = df[[selected_metric]].copy()
hist_data[selected_metric] = pd.to_numeric(
    hist_data[selected_metric],
    errors="coerce"
)
hist_data = hist_data.dropna()

fig_hist = px.histogram(
    hist_data,
    x=selected_metric,
    nbins=30,
    title=f"{korean_name(selected_metric)} 분포",
    labels={
        selected_metric: korean_name(selected_metric),
        "count": "선수 수"
    }
)

fig_hist.update_layout(
    xaxis_title=korean_name(selected_metric),
    yaxis_title="선수 수"
)

st.plotly_chart(fig_hist, use_container_width=True)


# --------------------------------------------------
# 9. 선수별 순위
# --------------------------------------------------

st.header("4. 선수별 타격 지표 순위")

rank_metric = st.selectbox(
    "순위를 확인할 항목을 선택하세요.",
    MAIN_COLS,
    format_func=korean_name,
    key="rank_metric"
)

rank_count = st.slider(
    "표시할 선수 수",
    min_value=5,
    max_value=30,
    value=10
)

rank_df = df[[NAME_COL, rank_metric]].copy()

rank_df[rank_metric] = pd.to_numeric(
    rank_df[rank_metric],
    errors="coerce"
)

rank_df = rank_df.dropna()

rank_df = rank_df.sort_values(
    by=rank_metric,
    ascending=False
).head(rank_count)

rank_df = rank_df.rename(
    columns={
        NAME_COL: "선수 이름",
        rank_metric: korean_name(rank_metric)
    }
)

st.dataframe(
    rank_df.reset_index(drop=True),
    use_container_width=True
)

fig_rank = px.bar(
    rank_df.sort_values(
        by=korean_name(rank_metric),
        ascending=True
    ),
    x=korean_name(rank_metric),
    y="선수 이름",
    orientation="h",
    title=f"{korean_name(rank_metric)} 상위 선수",
    text_auto=".2f"
)

st.plotly_chart(fig_rank, use_container_width=True)


# --------------------------------------------------
# 10. 전체 숫자형 열 상관관계 분석
# --------------------------------------------------

st.header("5. 전체 숫자형 데이터 상관관계 분석")

st.write(
    "원본 데이터의 모든 숫자형 열을 대상으로 "
    "피어슨 상관계수를 계산합니다."
)

corr_df = numeric_df.dropna(axis=1, how="all")

# 값이 모두 같은 열 제거
corr_df = corr_df.loc[
    :,
    corr_df.nunique(dropna=True) > 1
]

if corr_df.shape[1] >= 2:

    corr_matrix = corr_df.corr(method="pearson")

    korean_corr = corr_matrix.rename(
        index=korean_name,
        columns=korean_name
    )

    fig_corr = px.imshow(
        korean_corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="전체 숫자형 데이터 상관관계 히트맵"
    )

    fig_corr.update_layout(
        height=750
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    # 상관관계가 높은 항목 찾기
    upper_triangle = np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)

    corr_pairs = (
        corr_matrix.abs()
        .where(upper_triangle)
        .stack()
        .sort_values(ascending=False)
    )

    if not corr_pairs.empty:

        pair_rows = []

        for (col1, col2), abs_value in corr_pairs.items():
            pair_rows.append({
                "항목 1": korean_name(col1),
                "항목 2": korean_name(col2),
                "상관계수": corr_matrix.loc[col1, col2],
                "상관계수 절댓값": abs_value
            })

        pair_df = pd.DataFrame(pair_rows)

        st.subheader("상관관계가 높은 항목")

        st.dataframe(
            pair_df.head(15).round(3),
            use_container_width=True
        )

        strongest = pair_df.iloc[0]

        st.info(
            f"가장 강한 상관관계는 "
            f"'{strongest['항목 1']}'과 "
            f"'{strongest['항목 2']}' 사이에서 나타났으며, "
            f"상관계수는 {strongest['상관계수']:.3f}입니다."
        )

else:
    st.warning("상관관계를 계산할 숫자형 열이 부족합니다.")


# --------------------------------------------------
# 11. 주요 4개 항목 상관관계
# --------------------------------------------------

st.header("6. 주요 4개 지표 상관관계")

main_corr = numeric_df[MAIN_COLS].corr(method="pearson")

main_corr = main_corr.rename(
    index=korean_name,
    columns=korean_name
)

fig_main_corr = px.imshow(
    main_corr,
    text_auto=".3f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="주요 4개 지표 상관관계 히트맵"
)

st.plotly_chart(
    fig_main_corr,
    use_container_width=True
)


# --------------------------------------------------
# 12. 두 항목 선택 및 산점도
# --------------------------------------------------

st.header("7. 두 지표 사이의 관계 분석")

col1, col2 = st.columns(2)

with col1:
    x_col = st.selectbox(
        "독립변수(X축)",
        MAIN_COLS,
        index=0,
        format_func=korean_name
    )

with col2:
    y_col = st.selectbox(
        "종속변수(Y축)",
        MAIN_COLS,
        index=3,
        format_func=korean_name
    )

if x_col == y_col:
    st.warning("서로 다른 항목을 선택해 주세요.")

else:

    scatter_df = df[
        [NAME_COL, x_col, y_col]
    ].copy()

    scatter_df[x_col] = pd.to_numeric(
        scatter_df[x_col],
        errors="coerce"
    )

    scatter_df[y_col] = pd.to_numeric(
        scatter_df[y_col],
        errors="coerce"
    )

    scatter_df = scatter_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(scatter_df) >= 3:

        # 피어슨 상관계수
        correlation = scatter_df[x_col].corr(
            scatter_df[y_col],
            method="pearson"
        )

        # 선형 회귀 계산
        slope, intercept = np.polyfit(
            scatter_df[x_col],
            scatter_df[y_col],
            1
        )

        predicted = (
            slope * scatter_df[x_col] + intercept
        )

        # 결정계수 R²
        ss_res = np.sum(
            (scatter_df[y_col] - predicted) ** 2
        )

        ss_tot = np.sum(
            (scatter_df[y_col] - scatter_df[y_col].mean()) ** 2
        )

        if ss_tot != 0:
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = np.nan

        # 산점도
        fig_scatter = px.scatter(
            scatter_df,
            x=x_col,
            y=y_col,
            hover_name=NAME_COL,
            title=(
                f"{korean_name(x_col)}와 "
                f"{korean_name(y_col)}의 관계"
            ),
            labels={
                x_col: korean_name(x_col),
                y_col: korean_name(y_col)
            }
        )

        # 회귀 직선
        x_line = np.linspace(
            scatter_df[x_col].min(),
            scatter_df[x_col].max(),
            100
        )

        y_line = slope * x_line + intercept

        fig_scatter.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="선형 회귀 직선",
                line=dict(width=3)
            )
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

        # 분석 결과
        result1, result2, result3 = st.columns(3)

        result1.metric(
            "피어슨 상관계수",
            f"{correlation:.3f}"
        )

        result2.metric(
            "결정계수 R²",
            f"{r_squared:.3f}"
            if not np.isnan(r_squared)
            else "계산 불가"
        )

        result3.metric(
            "분석 데이터 수",
            f"{len(scatter_df):,}명"
        )

        st.subheader("선형 회귀 분석 결과")

        st.write(
            f"회귀식: "
            f"y = {slope:.4f}x + {intercept:.4f}"
        )

        st.write(
            f"상관계수: {correlation:.3f}"
        )

        st.write(
            f"결정계수: {r_squared:.3f}"
            if not np.isnan(r_squared)
            else "결정계수를 계산할 수 없습니다."
        )

        if correlation > 0:
            st.info(
                "두 항목은 양의 상관관계를 보입니다. "
                "한 항목이 증가할 때 다른 항목도 증가하는 경향이 있습니다."
            )
        elif correlation < 0:
            st.info(
                "두 항목은 음의 상관관계를 보입니다. "
                "한 항목이 증가할 때 다른 항목은 감소하는 경향이 있습니다."
            )
        else:
            st.info(
                "두 항목 사이에 선형 상관관계가 거의 나타나지 않습니다."
            )

        st.caption(
            "주의: 상관관계가 높다고 해서 한 항목이 다른 항목의 "
            "원인이라는 뜻은 아닙니다. 회귀분석은 두 변수 사이의 "
            "선형적인 관계를 설명하는 방법입니다."
        )

    else:
        st.warning(
            "분석에 사용할 수 있는 데이터가 충분하지 않습니다."
        )


# --------------------------------------------------
# 13. 데이터 분석 안내
# --------------------------------------------------

st.header("8. 분석 지표 설명")

explanations = {
    "배럴 타구 비율": (
        "배럴 타구가 전체 타구에서 차지하는 비율입니다. "
        "배럴은 타구 속도와 발사각이 조합된 질 좋은 타구를 의미합니다."
    ),
    "FB/LD 비율": (
        "플라이볼과 라인드라이브 타구의 비율을 나타내는 지표입니다."
    ),
    "95마일 이상 타구 비율": (
        "타구 속도가 시속 95마일 이상인 타구가 차지하는 비율입니다."
    ),
    "평균 타구 비거리": (
        "타구가 날아간 거리의 평균값입니다. 원본 데이터의 단위는 피트입니다."
    )
}

for name, explanation in explanations.items():
    st.markdown(f"**{name}**")
    st.write(explanation)


# --------------------------------------------------
# 14. 하단 안내
# --------------------------------------------------

st.divider()

st.caption(
    "데이터 출처: GitHub exit_velocity.csv | "
    "분석 방법: 기술통계, 피어슨 상관분석, 선형 회귀분석"
)

st.caption(
    "본 앱은 데이터 분석 학습을 위한 도구이며, "
    "상관관계가 인과관계를 의미하지는 않습니다."
)
