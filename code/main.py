
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 기본 설정
# -----------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "leeseojin09/baseball/refs/heads/main/"
    "data/exit_velocity.csv"
)

st.set_page_config(
    page_title="야구 타구 데이터 분석실",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ 야구 타구 데이터 기초 통계 분석실")
st.caption(
    "선수별 타구 속도 · 발사각 · 배럴 비율을 "
    "통계와 그래프로 분석합니다."
)

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 숫자형 열을 숫자로 변환
    for col in df.columns:
        if col not in ["last_name, first_name"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()

# -----------------------------
# 열 이름 설정
# -----------------------------
name_col = "last_name, first_name"

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

# 분석에 사용할 주요 변수
preferred_cols = [
    "avg_hit_angle",
    "avg_hit_speed",
    "max_hit_speed",
    "barrels",
    "brl_percent",
    "brl_pa",
    "anglesweetspotpercent",
    "avg_distance",
    "max_distance",
    "attempts"
]

available_cols = [
    c for c in preferred_cols if c in numeric_cols
]

# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.header("분석 설정")

if not numeric_cols:
    st.error("분석할 숫자형 데이터가 없습니다.")
    st.stop()

default_cols = available_cols or numeric_cols

selected_cols = st.sidebar.multiselect(
    "분석할 변수 선택",
    options=numeric_cols,
    default=default_cols
)

st.sidebar.caption(
    "변수 선택에 따라 통계표와 상관관계 분석이 변경됩니다."
)

# -----------------------------
# 데이터 개요
# -----------------------------
st.header("1. 데이터 개요")

c1, c2, c3, c4 = st.columns(4)

c1.metric("선수 데이터 수", f"{len(df):,}명")
c2.metric("전체 변수 수", f"{len(df.columns):,}개")
c3.metric(
    "결측치 개수",
    f"{int(df.isna().sum().sum()):,}개"
)
c4.metric(
    "중복 행 수",
    f"{int(df.duplicated().sum()):,}개"
)

with st.expander("원본 데이터 확인"):
    st.dataframe(df, use_container_width=True)

with st.expander("변수 설명"):
    descriptions = {
        "attempts": "타구 관련 시도 수",
        "avg_hit_angle": "평균 발사각",
        "max_hit_speed": "최고 타구 속도",
        "avg_hit_speed": "평균 타구 속도",
        "ev50": "타구 속도 중앙 구간 지표",
        "fbld": "뜬공·라인드라이브 관련 지표",
        "gb": "땅볼 관련 지표",
        "max_distance": "최대 타구 비거리",
        "avg_distance": "평균 타구 비거리",
        "barrels": "배럴 타구 수",
        "brl_percent": "배럴 타구 비율",
        "brl_pa": "타석당 배럴 비율 지표",
        "anglesweetspotpercent": "스위트 스폿 발사각 비율",
        "ev95plus": "시속 95마일 이상 타구 수",
        "ev95percent": "시속 95마일 이상 타구 비율"
    }

    for col in df.columns:
        st.write(
            f"**{col}**: "
            f"{descriptions.get(col, '원본 데이터의 변수')}"
        )

# -----------------------------
# 기초 통계
# -----------------------------
st.header("2. 기초 통계 분석")

if selected_cols:
    stats = df[selected_cols].describe().T
    stats["분산"] = df[selected_cols].var()
    stats["결측치"] = df[selected_cols].isna().sum()
    stats["중앙값"] = df[selected_cols].median()
    stats["왜도"] = df[selected_cols].skew()

    stats = stats.rename(columns={
        "count": "개수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "제1사분위수",
        "50%": "중앙값(50%)",
        "75%": "제3사분위수",
        "max": "최댓값"
    })

    st.dataframe(
        stats.round(3),
        use_container_width=True
    )

    st.download_button(
        "📥 기초 통계표 다운로드",
        data=stats.to_csv(
            encoding="utf-8-sig"
        ).encode("utf-8-sig"),
        file_name="baseball_basic_statistics.csv",
        mime="text/csv"
    )
else:
    st.info("사이드바에서 분석할 변수를 선택해 주세요.")

# -----------------------------
# 변수별 분포
# -----------------------------
st.header("3. 변수별 분포 분석")

if selected_cols:
    dist_col = st.selectbox(
        "분포를 확인할 변수",
        selected_cols
    )

    values = df[dist_col].dropna()

    if len(values) > 0:
        left, right = st.columns(2)

        with left:
            st.subheader("히스토그램")

            fig, ax = plt.subplots()
            ax.hist(
                values,
                bins=20,
                edgecolor="black"
            )
            ax.set_xlabel(dist_col)
            ax.set_ylabel("선수 수")
            ax.set_title(f"{dist_col} 분포")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("상자 수염 그래프")

            fig, ax = plt.subplots()
            ax.boxplot(values, vert=True)
            ax.set_ylabel(dist_col)
            ax.set_title(f"{dist_col} 상자 수염 그래프")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

        st.write(
            f"선택 변수의 평균은 **{values.mean():.3f}**, "
            f"중앙값은 **{values.median():.3f}**, "
            f"표준편차는 **{values.std():.3f}**입니다."
        )

# -----------------------------
# 선수별 비교
# -----------------------------
st.header("4. 선수별 비교")

if name_col in df.columns and selected_cols:
    compare_col = st.selectbox(
        "비교 기준",
        selected_cols,
        key="compare_col"
    )

    top_n = st.slider(
        "표시할 선수 수",
        min_value=5,
        max_value=30,
        value=10
    )

    ranking = df[[name_col, compare_col]].dropna()
    ranking = ranking.sort_values(
        compare_col,
        ascending=False
    ).head(top_n)

    st.dataframe(
        ranking.reset_index(drop=True),
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, max(4, top_n * 0.3))
    )
    ax.barh(
        ranking[name_col].iloc[::-1],
        ranking[compare_col].iloc[::-1]
    )
    ax.set_xlabel(compare_col)
    ax.set_ylabel("선수")
    ax.set_title(f"{compare_col} 상위 {top_n}명")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    player_query = st.text_input(
        "선수 이름 검색",
        placeholder="예: Ohtani"
    )

    if player_query:
        result = df[
            df[name_col].astype(str).str.contains(
                player_query,
                case=False,
                na=False
            )
        ]
        st.dataframe(result, use_container_width=True)

# -----------------------------
# 상관관계 분석
# -----------------------------
st.header("5. 상관관계 분석")

corr_candidates = [
    c for c in [
        "avg_hit_angle",
        "avg_hit_speed",
        "max_hit_speed",
        "barrels",
        "brl_percent",
        "brl_pa",
        "avg_distance",
        "max_distance",
        "anglesweetspotpercent"
    ]
    if c in numeric_cols
]

if len(corr_candidates) >= 2:
    corr_cols = st.multiselect(
        "상관관계를 분석할 변수",
        corr_candidates,
        default=corr_candidates
    )

    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(method="pearson")

        st.subheader("피어슨 상관계수 행렬")
        st.dataframe(
            corr.round(3),
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(max(6, len(corr_cols)), 
                     max(4, len(corr_cols) * 0.7))
        )

        image = ax.imshow(
            corr,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            aspect="auto"
        )

        ax.set_xticks(range(len(corr_cols)))
        ax.set_xticklabels(
            corr_cols, rotation=45, ha="right"
        )
        ax.set_yticks(range(len(corr_cols)))
        ax.set_yticklabels(corr_cols)

        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=9
                    )

        fig.colorbar(image, ax=ax, label="상관계수")
        ax.set_title("변수 간 상관관계")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "상관계수는 -1부터 1까지이며, "
            "0에 가까울수록 선형 관계가 약합니다. "
            "상관관계가 인과관계를 의미하지는 않습니다."
        )

        if (
            "avg_hit_angle" in corr_cols
            and "avg_hit_speed" in corr_cols
        ):
            r = corr.loc["avg_hit_angle", "avg_hit_speed"]
            st.metric(
                "평균 발사각과 평균 타구 속도의 상관계수",
                f"{r:.3f}" if pd.notna(r) else "계산 불가"
            )
else:
    st.info("상관관계 분석에 필요한 숫자형 변수가 부족합니다.")

# -----------------------------
# 발사각과 타구 속도 산점도
# -----------------------------
st.header("6. 발사각과 타구 속도 비교")

angle_col = "avg_hit_angle"
speed_col = "avg_hit_speed"

if angle_col in df.columns and speed_col in df.columns:
    scatter_data = df[
        [name_col, angle_col, speed_col]
        if name_col in df.columns
        else [angle_col, speed_col]
    ].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        scatter_data[angle_col],
        scatter_data[speed_col],
        alpha=0.65
    )
    ax.set_xlabel("평균 발사각")
    ax.set_ylabel("평균 타구 속도")
    ax.set_title("선수별 평균 발사각과 평균 타구 속도")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "각 점은 개별 타구가 아니라 한 선수의 평균 기록을 나타냅니다."
    )

# -----------------------------
# 원본 데이터 다운로드
# -----------------------------
st.header("7. 데이터 다운로드")

st.download_button(
    "📥 분석 데이터 CSV 다운로드",
    data=df.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig"),
    file_name="exit_velocity_analysis.csv",
    mime="text/csv"
)

st.success("분석이 완료되었습니다.")
