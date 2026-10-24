import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
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
    "실제 CSV 데이터를 이용해 타구 속도의 분포, "
    "기초 통계량, 선수별 기록, 변수 간 상관관계를 분석합니다."
)


# -----------------------------------
# 2. 데이터 불러오기
# -----------------------------------
@st.cache_data(ttl=3600)
def load_data():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    # UTF-8 형식으로 CSV 읽기
    df = pd.read_csv(
        BytesIO(response.content),
        encoding="utf-8-sig"
    )

    # 열 이름 앞뒤의 공백 제거
    df.columns = df.columns.astype(str).str.strip()

    # 완전히 비어 있는 열 제거
    df = df.dropna(axis=1, how="all")

    # 숫자로 변환할 수 있는 열만 숫자로 변환
    # 선수 이름 등 문자 데이터는 그대로 유지
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
    st.write("확인할 CSV 주소:")
    st.code(DATA_URL)
    st.info(
        "GitHub CSV 주소와 파일 경로를 확인하고, "
        "requirements.txt에 requests가 있는지 확인해 주세요."
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
    st.metric(
        "결측값 수",
        f"{int(df.isna().sum().sum()):,}개"
    )

with st.expander("데이터 미리 보기", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("전체 열 이름"):
    st.write(list(df.columns))

with st.expander("열별 데이터 유형 및 결측값"):
    info_df = pd.DataFrame({
        "열 이름": df.columns,
        "데이터 유형": [
            str(df[col].dtype) for col in df.columns
        ],
        "결측값 수": [
            int(df[col].isna().sum()) for col in df.columns
        ],
        "결측 비율(%)": [
            round(df[col].isna().mean() * 100, 2)
            for col in df.columns
        ]
    })
    st.dataframe(info_df, use_container_width=True)


# -----------------------------------
# 4. 숫자형 변수 선택
# -----------------------------------
numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

if not numeric_cols:
    st.error(
        "분석할 숫자형 변수가 없습니다. "
        "CSV의 열 이름과 데이터 형식을 확인해 주세요."
    )
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

    st.dataframe(
        stats.round(3),
        use_container_width=True
    )
else:
    st.info("통계량을 확인할 변수를 선택해 주세요.")


# -----------------------------------
# 6. 히스토그램
# -----------------------------------
st.subheader("3. 데이터 분포 분석 — 히스토그램")

hist_col = st.selectbox(
    "히스토그램으로 확인할 변수",
    options=numeric_cols,
    index=(
        numeric_cols.index("avg_hit_speed")
        if "avg_hit_speed" in numeric_cols
        else 0
    )
)

hist_data = df[hist_col].dropna()

if len(hist_data) > 0:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.hist(
        hist_data,
        bins=20,
        edgecolor="black"
    )

    ax.set_title(f"{hist_col} 분포")
    ax.set_xlabel(hist_col)
    ax.set_ylabel("데이터 개수")
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.write(f"유효 데이터 수: {len(hist_data):,}개")
    st.write(f"평균: {hist_data.mean():.3f}")
    st.write(f"중앙값: {hist_data.median():.3f}")
else:
    st.warning("선택한 변수에 분석할 값이 없습니다.")


# -----------------------------------
# 7. 상자그림
# -----------------------------------
st.subheader("4. 이상값과 분포 분석 — 상자그림")

box_cols = st.multiselect(
    "상자그림으로 비교할 변수",
    options=numeric_cols,
    default=(
        ["avg_hit_speed"]
        if "avg_hit_speed" in numeric_cols
        else numeric_cols[:1]
    )
)

if box_cols:
    box_data = [
        df[col].dropna().values
        for col in box_cols
    ]

    valid_pairs = [
        (col, values)
        for col, values in zip(box_cols, box_data)
        if len(values) > 0
    ]

    if valid_pairs:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.boxplot(
            [values for _, values in valid_pairs],
            tick_labels=[col for col, _ in valid_pairs],
            showfliers=True
        )

        ax.set_title("변수별 상자그림")
        ax.set_ylabel("값")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "상자그림의 점은 일반적인 기준에서 벗어난 "
            "이상값 후보입니다. 반드시 오류 데이터라는 뜻은 아닙니다."
        )
    else:
        st.warning("선택한 변수에 분석할 값이 없습니다.")
else:
    st.info("비교할 변수를 하나 이상 선택해 주세요.")


# -----------------------------------
# 8. 선수별 기록 순위
# -----------------------------------
st.subheader("5. 선수별 기록 순위")

# 선수 이름 열 찾기
possible_name_cols = [
    "last_name, first_name",
    "player_name",
    "player",
    "name",
    "선수명",
    "선수 이름"
]

name_col = next(
    (col for col in possible_name_cols if col in df.columns),
    None
)

if name_col is None:
    st.warning(
        "선수 이름 열을 자동으로 찾지 못했습니다. "
        "아래에서 선수 이름 열을 직접 선택할 수 있습니다."
    )

    text_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if text_cols:
        name_col = st.selectbox(
            "선수 이름이 들어 있는 열 선택",
            options=text_cols
        )

if name_col is not None:
    rank_col = st.selectbox(
        "순위를 확인할 기록",
        options=numeric_cols,
        index=(
            numeric_cols.index("avg_hit_speed")
            if "avg_hit_speed" in numeric_cols
            else 0
        )
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

    # 같은 선수의 기록이 여러 행이면 평균으로 집계
    rank_df = (
        rank_df.groupby(name_col, as_index=False)[rank_col]
        .mean()
    )

    rank_df = rank_df.sort_values(
        rank_col,
        ascending=(rank_order == "낮은 순")
    ).head(rank_count)

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

    if not rank_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            rank_df[name_col].astype(str).iloc[::-1],
            rank_df[rank_col].iloc[::-1]
        )

        ax.set_title(f"선수별 {rank_col} 순위")
        ax.set_xlabel(rank_col)
        ax.set_ylabel("선수")
        ax.grid(axis="x", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)
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
    st.dataframe(
        corr_data.round(3),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(corr_cols) * 1.1),
            max(5, len(corr_cols) * 0.8)
        )
    )

    image = ax.imshow(
        corr_data,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            value = corr_data.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

    ax.set_title("상관관계 히트맵")
    fig.colorbar(image, ax=ax, label="상관계수")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "상관계수는 -1부터 1 사이의 값입니다. "
        "절댓값이 1에 가까울수록 선형 관계가 강하고, "
        "0에 가까울수록 선형 관계가 약합니다. "
        "상관관계만으로 인과관계를 판단할 수는 없습니다."
    )
else:
    st.info("상관관계를 분석하려면 변수를 2개 이상 선택해 주세요.")


# -----------------------------------
# 10. 산점도
# -----------------------------------
st.subheader("7. 두 변수의 관계 — 산점도")

scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    x_col = st.selectbox(
        "X축 변수",
        options=numeric_cols,
        index=(
            numeric_cols.index("avg_hit_speed")
            if "avg_hit_speed" in numeric_cols
            else 0
        ),
        key="scatter_x"
    )

with scatter_col2:
    default_y_index = (
        1 if len(numeric_cols) > 1 else 0
    )

    y_col = st.selectbox(
        "Y축 변수",
        options=numeric_cols,
        index=default_y_index,
        key="scatter_y"
    )

scatter_df = df[[x_col, y_col]].dropna()

if len(scatter_df) >= 2:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        scatter_df[x_col],
        scatter_df[y_col],
        alpha=0.6
    )

    ax.set_title(f"{x_col}와 {y_col}의 관계")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)

    # 단순 선형 추세선
    if (
        x_col != y_col
        and scatter_df[x_col].nunique() >= 2
        and scatter_df[y_col].nunique() >= 2
    ):
        slope, intercept = np.polyfit(
            scatter_df[x_col],
            scatter_df[y_col],
            1
        )

        x_values = np.linspace(
            scatter_df[x_col].min(),
            scatter_df[x_col].max(),
            100
        )

        ax.plot(
            x_values,
            slope * x_values + intercept,
            linestyle="--",
            label="선형 추세선"
        )

        ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    if x_col != y_col:
        corr_value = scatter_df[x_col].corr(
            scatter_df[y_col]
        )

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "피어슨 상관계수",
                f"{corr_value:.4f}"
                if pd.notna(corr_value)
                else "계산 불가"
            )

        with metric_col2:
            st.metric(
                "분석에 사용된 데이터",
                f"{len(scatter_df):,}개"
            )
else:
    st.warning(
        "산점도를 그리기 위한 유효한 데이터가 부족합니다."
    )


# -----------------------------------
# 11. CSV 다운로드
# -----------------------------------
st.subheader("8. 분석 데이터 다운로드")

csv_data = df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="📥 분석 데이터 CSV 다운로드",
    data=csv_data,
    file_name="exit_velocity_analysis.csv",
    mime="text/csv"
)

st.success("데이터 분석 화면을 모두 불러왔습니다.")
st.caption(
    "분석 결과는 원본 CSV에 포함된 데이터와 열에 따라 달라집니다."
)
