
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from xgboost import XGBRegressor


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="야구 데이터 분석",
    page_icon="⚾",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "leeseojin09/baseball/refs/heads/main/"
    "data/exit_velocity.csv"
)

NAME_COL = "last_name, first_name"

MAIN_COLS = [
    "brl_percent",
    "fbld",
    "ev95percent",
    "avg_distance",
    "avg_hit_angle"
]

KOREAN_NAMES = {
    "last_name, first_name": "선수 이름",
    "player_id": "선수 ID",
    "attempts": "타구 데이터 수",
    "avg_hit_angle": "평균 발사각",
    "anglesweetspotpercent": "이상적인 발사각 비율",
    "max_hit_speed": "최고 타구 속도",
    "avg_hit_speed": "평균 타구 속도",
    "ev50": "타구 속도 중앙값",
    "fbld": "뜬공·직선타 비율",
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


def korean_name(col):
    return KOREAN_NAMES.get(col, col)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():
    response = requests.get(DATA_URL, timeout=20)
    response.raise_for_status()

    from io import BytesIO

    data = pd.read_csv(BytesIO(response.content))

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# =========================================================
# 3. 제목 및 소개
# =========================================================

st.title("⚾ 야구 타격 데이터 분석실")

st.write(
    "메이저리그 타격 데이터를 활용하여 "
    "배럴 타구 비율, 타구 속도, 발사각, 비거리 등을 분석합니다."
)

st.divider()


# =========================================================
# 4. 데이터 기본 정보
# =========================================================

st.header("📊 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 선수 및 데이터 행 수", f"{len(df):,}개")

with col2:
    st.metric("전체 변수 수", f"{len(df.columns):,}개")

with col3:
    st.metric(
        "분석 가능한 숫자형 변수",
        f"{len(df.select_dtypes(include=np.number).columns):,}개"
    )

st.subheader("전체 원본 데이터")

display_df = df.rename(columns=korean_name)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

csv_data = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "📥 원본 데이터 CSV 다운로드",
    data=csv_data,
    file_name="baseball_data.csv",
    mime="text/csv"
)

st.divider()


# =========================================================
# 5. 주요 타격 지표 분석
# =========================================================

st.header("⚾ 주요 타격 지표 분석")

missing_cols = [col for col in MAIN_COLS if col not in df.columns]

if missing_cols:
    st.error(
        "CSV에 다음 분석 열이 없습니다: "
        + ", ".join(missing_cols)
    )
    st.stop()


analysis_df = df.copy()

for col in MAIN_COLS:
    analysis_df[col] = pd.to_numeric(
        analysis_df[col],
        errors="coerce"
    )


# 주요 지표 카드
metric_cols = st.columns(5)

for i, col in enumerate(MAIN_COLS):
    value = analysis_df[col].mean()

    with metric_cols[i]:
        st.metric(
            korean_name(col),
            f"{value:.2f}" if pd.notna(value) else "자료 없음"
        )


# =========================================================
# 6. 기술통계
# =========================================================

st.subheader("1. 주요 지표 기술통계")

stats_df = analysis_df[MAIN_COLS].describe().T

stats_df = stats_df.rename(
    index=korean_name,
    columns={
        "count": "개수",
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
    stats_df.round(3),
    use_container_width=True
)


# =========================================================
# 7. 히스토그램
# =========================================================

st.subheader("2. 타격 지표 분포")

selected_hist = st.selectbox(
    "분포를 확인할 지표 선택",
    MAIN_COLS,
    format_func=korean_name
)

hist_data = analysis_df[selected_hist].dropna()

fig = px.histogram(
    x=hist_data,
    nbins=30,
    title=f"{korean_name(selected_hist)} 분포",
    labels={"x": korean_name(selected_hist), "count": "선수 수"}
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 8. 선수별 순위
# =========================================================

st.subheader("3. 선수별 타격 지표 순위")

rank_col = st.selectbox(
    "순위를 확인할 지표",
    MAIN_COLS,
    format_func=korean_name
)

rank_count = st.slider(
    "표시할 선수 수",
    min_value=5,
    max_value=30,
    value=10
)

rank_df = df[[NAME_COL, rank_col]].copy()

rank_df[rank_col] = pd.to_numeric(
    rank_df[rank_col],
    errors="coerce"
)

rank_df = rank_df.dropna(
    subset=[rank_col]
).sort_values(
    rank_col,
    ascending=False
).head(rank_count)

rank_df = rank_df.rename(
    columns={
        NAME_COL: "선수 이름",
        rank_col: korean_name(rank_col)
    }
)

st.dataframe(
    rank_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 9. 전체 숫자형 변수 상관관계 분석
# =========================================================

st.divider()

st.header("🔍 전체 숫자형 변수 상관관계 분석")

numeric_df = df.select_dtypes(
    include=np.number
).copy()

# 실제 숫자로 변환 가능한 문자열 열도 포함
for col in df.columns:
    if col not in numeric_df.columns:
        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().any():
            numeric_df[col] = converted


if numeric_df.shape[1] >= 2:

    corr_all = numeric_df.corr(method="pearson")

    st.subheader("1. 전체 변수 상관관계 히트맵")

    fig = px.imshow(
        corr_all,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="전체 숫자형 변수 피어슨 상관관계"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("2. 상관관계 계수표")

    corr_display = corr_all.rename(
        index=korean_name,
        columns=korean_name
    )

    st.dataframe(
        corr_display.round(3),
        use_container_width=True
    )

    st.subheader("3. 상관관계가 높은 변수 쌍")

    corr_pairs = corr_all.where(
        np.triu(
            np.ones(corr_all.shape),
            k=1
        ).astype(bool)
    ).stack().reset_index()

    corr_pairs.columns = [
        "항목 1",
        "항목 2",
        "상관계수"
    ]

    corr_pairs["절댓값"] = corr_pairs["상관계수"].abs()

    corr_pairs = corr_pairs.sort_values(
        "절댓값",
        ascending=False
    )

    corr_pairs_display = corr_pairs.drop(
        columns=["절댓값"]
    ).copy()

    corr_pairs_display["항목 1"] = (
        corr_pairs_display["항목 1"].map(korean_name)
    )

    corr_pairs_display["항목 2"] = (
        corr_pairs_display["항목 2"].map(korean_name)
    )

    st.dataframe(
        corr_pairs_display.round(3),
        use_container_width=True,
        hide_index=True
    )

    if not corr_pairs.empty:
        strongest = corr_pairs.iloc[0]

        st.info(
            f"가장 강한 상관관계는 "
            f"'{korean_name(strongest['항목 1'])}'과 "
            f"'{korean_name(strongest['항목 2'])}' 사이에서 나타났으며, "
            f"상관계수는 {strongest['상관계수']:.3f}입니다."
        )

else:
    st.warning("상관관계를 계산할 숫자형 변수가 부족합니다.")


# =========================================================
# 10. 주요 5개 변수 상관관계 분석
# =========================================================

st.divider()

st.header("📈 주요 5개 타격 지표 상관관계")

corr_main = analysis_df[MAIN_COLS].corr()

corr_main_display = corr_main.rename(
    index=korean_name,
    columns=korean_name
)

fig = px.imshow(
    corr_main_display,
    text_auto=".3f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="주요 타격 지표 피어슨 상관관계"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 11. 직접 계산하는 선형회귀 분석
# =========================================================

st.divider()

st.header("📉 주요 타격 지표 선형 관계 분석")

st.write(
    "두 타격 지표 사이의 관계를 산점도와 회귀직선으로 확인합니다."
)

reg_col1, reg_col2 = st.columns(2)

with reg_col1:
    x_col = st.selectbox(
        "독립변수(X)",
        MAIN_COLS,
        index=4,
        format_func=korean_name,
        key="basic_x"
    )

with reg_col2:
    y_col = st.selectbox(
        "종속변수(Y)",
        MAIN_COLS,
        index=0,
        format_func=korean_name,
        key="basic_y"
    )

if x_col == y_col:
    st.warning("서로 다른 변수를 선택해 주세요.")

else:
    reg_data = analysis_df[[x_col, y_col]].dropna()

    if len(reg_data) >= 2:
        x = reg_data[x_col].values
        y = reg_data[y_col].values

        slope, intercept = np.polyfit(x, y, 1)

        predicted = slope * x + intercept

        corr_value = np.corrcoef(x, y)[0, 1]
        r_squared = corr_value ** 2

        fig = px.scatter(
            reg_data,
            x=x_col,
            y=y_col,
            labels={
                x_col: korean_name(x_col),
                y_col: korean_name(y_col)
            },
            title="타격 지표 산점도와 회귀직선"
        )

        line_x = np.linspace(
            x.min(),
            x.max(),
            100
        )

        line_y = slope * line_x + intercept

        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="회귀직선"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("피어슨 상관계수", f"{corr_value:.4f}")
        c2.metric("결정계수 R²", f"{r_squared:.4f}")
        c3.metric("분석 데이터 수", f"{len(reg_data)}개")

        st.write(
            f"**회귀식:** "
            f"Y = {slope:.4f}X + {intercept:.4f}"
        )

    else:
        st.warning("회귀분석에 필요한 데이터가 부족합니다.")


# =========================================================
# 12. 머신러닝 분석
# =========================================================


# =========================================================
# 12. 머신러닝 분석
# =========================================================

st.divider()
st.header("🤖 머신러닝 분석")

st.write(
    "분석 방법을 선택하면 해당 머신러닝 모델을 실행합니다."
)

# 분석 방법 선택
model_type = st.selectbox(
    "분석 방법 선택",
    [
        "다중 선형회귀",
        "랜덤 포레스트",
        "XGBoost"
    ],
    key="model_type"
)

st.subheader(f"📌 {model_type} 분석")

# 종속변수 선택
target_col = st.selectbox(
    "예측할 종속변수",
    MAIN_COLS,
    format_func=korean_name,
    key="ml_target"
)

# 나머지 4개 변수를 독립변수로 사용
feature_cols = [
    col for col in MAIN_COLS
    if col != target_col
]

st.write("**독립변수:**")
st.write(", ".join(korean_name(c) for c in feature_cols))

st.write("**종속변수:**", korean_name(target_col))


# =========================================================
# 데이터 전처리
# =========================================================

ml_data = analysis_df[
    feature_cols + [target_col]
].copy()

ml_data = ml_data.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

if len(ml_data) < 10:
    st.warning("분석에 필요한 데이터가 부족합니다.")

else:

    X = ml_data[feature_cols]
    y = ml_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================================
    # 모델 선택 및 학습
    # =====================================================

    if model_type == "다중 선형회귀":

        model = LinearRegression()

    elif model_type == "랜덤 포레스트":

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )

    else:

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0
        )

    # 학습
    model.fit(X_train, y_train)

    # 예측
    predictions = model.predict(X_test)

    # =====================================================
    # 성능 평가
    # =====================================================

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    st.subheader("1. 모델 성능 평가")

    c1, c2, c3 = st.columns(3)

    c1.metric("결정계수 R²", f"{r2:.4f}")
    c2.metric("평균 절대 오차 MAE", f"{mae:.4f}")
    c3.metric("평균 제곱근 오차 RMSE", f"{rmse:.4f}")

    st.caption(
        "R²는 높을수록 좋고 MAE와 RMSE는 낮을수록 좋습니다."
    )

    # =====================================================
    # 실제값과 예측값 비교
    # =====================================================

    st.subheader("2. 실제값과 예측값 비교")

    comparison_df = pd.DataFrame({
        "실제값": y_test.values,
        "예측값": predictions
    })

    fig = px.scatter(
        comparison_df,
        x="실제값",
        y="예측값",
        title=f"{model_type} 실제값과 예측값"
    )

    min_value = min(
        comparison_df["실제값"].min(),
        comparison_df["예측값"].min()
    )

    max_value = max(
        comparison_df["실제값"].max(),
        comparison_df["예측값"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="이상적인 예측선"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # 모델별 추가 분석
    # =====================================================

    if model_type == "다중 선형회귀":

        st.subheader("3. 회귀계수 분석")

        coefficient_df = pd.DataFrame({
            "독립변수": [
                korean_name(c) for c in feature_cols
            ],
            "회귀계수": model.coef_
        })

        st.dataframe(
            coefficient_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"**절편:** {model.intercept_:.4f}"
        )

        st.info(
            "회귀계수는 다른 독립변수가 일정하다고 가정할 때 "
            "각 독립변수가 종속변수 예측값에 갖는 선형 관계를 나타냅니다."
        )

    elif model_type in ["랜덤 포레스트", "XGBoost"]:

        st.subheader("3. 변수 중요도 분석")

        importance_df = pd.DataFrame({
            "변수": [
                korean_name(c) for c in feature_cols
            ],
            "중요도": model.feature_importances_
        }).sort_values(
            "중요도",
            ascending=False
        )

        fig = px.bar(
            importance_df,
            x="중요도",
            y="변수",
            orientation="h",
            title=f"{model_type} 변수 중요도"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            importance_df.round(4),
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # 결과 다운로드
    # =====================================================

    st.subheader("4. 예측 결과 다운로드")

    result_df = comparison_df.copy()

    result_df.insert(
        0,
        "분석 방법",
        model_type
    )

    result_df.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "📥 예측 결과 CSV 다운로드",
        data=result_df.to_csv(
            index=False,
            encoding="utf-8-sig"
        ),
        file_name=f"{model_type}_예측결과.csv",
        mime="text/csv"
    )


# =========================================================
# 마무리
# =========================================================

st.divider()

st.caption(
    "야구 타격 데이터 분석실 | 데이터 분석 및 머신러닝"
)

