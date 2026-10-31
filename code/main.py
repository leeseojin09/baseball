
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO

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
    page_title="야구 데이터 분석실",
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
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


missing_cols = [
    col for col in MAIN_COLS
    if col not in df.columns
]

if missing_cols:
    st.error(
        "CSV에 필요한 열이 없습니다: "
        + ", ".join(missing_cols)
    )
    st.stop()


# =========================================================
# 3. 사이드바 메뉴
# =========================================================

st.sidebar.title("⚾ 야구 데이터 분석실")
st.sidebar.caption("메뉴를 선택하여 분석을 진행하세요.")

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "데이터 탐색",
        "데이터 전처리",
        "모델 학습",
        "학습 결과"
    ]
)

st.sidebar.divider()

st.sidebar.write("**분석 모델**")
st.sidebar.write("• 다중 선형회귀")
st.sidebar.write("• 랜덤 포레스트")
st.sidebar.write("• XGBoost")


# =========================================================
# 4. 공통 전처리 데이터
# =========================================================

numeric_df = df.copy()

for col in numeric_df.columns:
    converted = pd.to_numeric(
        numeric_df[col],
        errors="coerce"
    )

    if converted.notna().any():
        numeric_df[col] = converted


# =========================================================
# 메뉴 1. 데이터 탐색
# =========================================================

if menu == "데이터 탐색":

    st.title("📊 데이터 탐색")

    st.write(
        "메이저리그 타격 데이터를 확인하고 "
        "주요 타격 지표의 통계적 관계를 분석합니다."
    )

    # 데이터 기본 정보
    c1, c2, c3 = st.columns(3)

    c1.metric("전체 데이터", f"{len(df):,}개")
    c2.metric("전체 변수", f"{len(df.columns)}개")
    c3.metric(
        "숫자형 변수",
        f"{len(numeric_df.select_dtypes(include=np.number).columns)}개"
    )

    st.divider()

    # 원본 데이터
    st.subheader("1. 전체 원본 데이터")

    display_df = df.rename(columns=korean_name)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 원본 데이터 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="baseball_data.csv",
        mime="text/csv"
    )

    st.divider()

    # 주요 지표
    st.subheader("2. 주요 타격 지표")

    metric_cols = st.columns(5)

    for i, col in enumerate(MAIN_COLS):
        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        with metric_cols[i]:
            value = values.mean()
            st.metric(
                korean_name(col),
                f"{value:.2f}" if pd.notna(value) else "자료 없음"
            )

    # 기술통계
    st.subheader("3. 주요 지표 기술통계")

    stats = numeric_df[MAIN_COLS].describe().T

    stats = stats.rename(
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
        stats.round(3),
        use_container_width=True
    )

    st.divider()

    # 히스토그램
    st.subheader("4. 타격 지표 분포")

    hist_col = st.selectbox(
        "분포를 확인할 변수",
        MAIN_COLS,
        format_func=korean_name
    )

    hist_data = numeric_df[hist_col].dropna()

    fig = px.histogram(
        x=hist_data,
        nbins=30,
        title=f"{korean_name(hist_col)} 분포",
        labels={
            "x": korean_name(hist_col),
            "count": "데이터 수"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 선수별 순위
    st.subheader("5. 선수별 순위")

    rank_col = st.selectbox(
        "순위를 확인할 지표",
        MAIN_COLS,
        format_func=korean_name
    )

    rank_count = st.slider(
        "표시할 선수 수",
        5, 30, 10
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

    st.divider()

    # 전체 숫자형 변수 상관관계
    st.subheader("6. 전체 숫자형 변수 상관관계")

    all_numeric = numeric_df.select_dtypes(
        include=np.number
    )

    if all_numeric.shape[1] >= 2:

        corr = all_numeric.corr()

        corr_display = corr.rename(
            index=korean_name,
            columns=korean_name
        )

        fig = px.imshow(
            corr_display,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="전체 변수 피어슨 상관관계"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        pairs = corr.where(
            np.triu(
                np.ones(corr.shape),
                k=1
            ).astype(bool)
        ).stack().reset_index()

        pairs.columns = [
            "항목 1",
            "항목 2",
            "상관계수"
        ]

        pairs["절댓값"] = pairs["상관계수"].abs()

        pairs = pairs.sort_values(
            "절댓값",
            ascending=False
        )

        pairs_display = pairs.drop(
            columns=["절댓값"]
        ).copy()

        pairs_display["항목 1"] = (
            pairs_display["항목 1"].map(korean_name)
        )

        pairs_display["항목 2"] = (
            pairs_display["항목 2"].map(korean_name)
        )

        st.subheader("상관관계가 높은 변수 쌍")

        st.dataframe(
            pairs_display.round(3),
            use_container_width=True,
            hide_index=True
        )

    # 주요 5개 변수 회귀 분석
    st.divider()
    st.subheader("7. 주요 지표 회귀 분석")

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox(
            "독립변수",
            MAIN_COLS,
            index=4,
            format_func=korean_name
        )

    with col2:
        y_col = st.selectbox(
            "종속변수",
            MAIN_COLS,
            index=0,
            format_func=korean_name
        )

    if x_col == y_col:
        st.warning("서로 다른 변수를 선택해 주세요.")

    else:
        reg_data = numeric_df[[x_col, y_col]].dropna()

        if len(reg_data) >= 2:

            x = reg_data[x_col].values
            y = reg_data[y_col].values

            slope, intercept = np.polyfit(x, y, 1)

            corr_value = np.corrcoef(x, y)[0, 1]

            fig = px.scatter(
                reg_data,
                x=x_col,
                y=y_col,
                labels={
                    x_col: korean_name(x_col),
                    y_col: korean_name(y_col)
                },
                title="산점도와 회귀직선"
            )

            line_x = np.linspace(
                x.min(),
                x.max(),
                100
            )

            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=slope * line_x + intercept,
                    mode="lines",
                    name="회귀직선"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.write(
                f"회귀식: Y = {slope:.4f}X + {intercept:.4f}"
            )

            st.metric(
                "피어슨 상관계수",
                f"{corr_value:.4f}"
            )


# =========================================================
# 메뉴 2. 데이터 전처리
# =========================================================

elif menu == "데이터 전처리":

    st.title("🧹 데이터 전처리")

    st.write(
        "머신러닝 학습에 사용될 데이터를 확인하고 "
        "결측치와 자료형을 처리합니다."
    )

    st.subheader("1. 전처리 전 데이터")

    st.write(f"전체 행: {len(df)}개")
    st.write(f"전체 열: {len(df.columns)}개")

    st.dataframe(
        df.head(20).rename(columns=korean_name),
        use_container_width=True
    )

    st.divider()

    # 자료형 확인
    st.subheader("2. 데이터 자료형")

    dtype_df = pd.DataFrame({
        "변수": df.columns,
        "자료형": df.dtypes.astype(str).values,
        "결측치 수": df.isna().sum().values
    })

    dtype_df["변수"] = dtype_df["변수"].map(korean_name)

    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # 결측치 확인
    st.subheader("3. 결측치 분석")

    missing_df = pd.DataFrame({
        "변수": df.columns,
        "결측치 수": df.isna().sum().values,
        "결측치 비율(%)": (
            df.isna().mean().values * 100
        )
    })

    missing_df["변수"] = missing_df["변수"].map(korean_name)

    st.dataframe(
        missing_df.sort_values(
            "결측치 수",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # 주요 분석 변수 설정
    st.divider()
    st.subheader("4. 머신러닝 전처리 설정")

    target_col = st.selectbox(
        "예측할 종속변수",
        MAIN_COLS,
        format_func=korean_name,
        key="prep_target"
    )

    feature_cols = [
        col for col in MAIN_COLS
        if col != target_col
    ]

    st.write("독립변수:")
    st.write(", ".join(korean_name(c) for c in feature_cols))

    st.write("종속변수:")
    st.write(korean_name(target_col))

    # 숫자형 변환
    selected_cols = feature_cols + [target_col]

    prepared_df = df[selected_cols].copy()

    for col in selected_cols:
        prepared_df[col] = pd.to_numeric(
            prepared_df[col],
            errors="coerce"
        )

    prepared_df = prepared_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    before_rows = len(prepared_df)

    # 결측치 제거
    prepared_df = prepared_df.dropna()

    after_rows = len(prepared_df)

    removed_rows = before_rows - after_rows

    st.subheader("5. 전처리 결과")

    c1, c2, c3 = st.columns(3)

    c1.metric("전처리 전 행 수", before_rows)
    c2.metric("전처리 후 행 수", after_rows)
    c3.metric("제거된 행 수", removed_rows)

    st.write("전처리된 데이터")

    display_prepared = prepared_df.rename(
        columns=korean_name
    )

    st.dataframe(
        display_prepared,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 전처리 데이터 다운로드",
        data=prepared_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="preprocessed_data.csv",
        mime="text/csv"
    )

    # 세션에 전처리 데이터 저장
    st.session_state["prepared_df"] = prepared_df
    st.session_state["prepared_target"] = target_col
    st.session_state["prepared_features"] = feature_cols

    st.success(
        "전처리가 완료되었습니다. 모델 학습 메뉴에서 학습을 진행하세요."
    )


# =========================================================
# 메뉴 3. 모델 학습
# =========================================================

elif menu == "모델 학습":

    st.title("🤖 모델 학습")

    st.write(
        "전처리된 데이터를 활용하여 머신러닝 모델을 학습합니다."
    )

    # 종속변수 선택
    target_col = st.selectbox(
        "예측할 종속변수",
        MAIN_COLS,
        format_func=korean_name,
        key="train_target"
    )

    feature_cols = [
        col for col in MAIN_COLS
        if col != target_col
    ]

    st.subheader("1. 학습 데이터 구성")

    st.write("독립변수:")
    st.write(", ".join(korean_name(c) for c in feature_cols))

    st.write("종속변수:")
    st.write(korean_name(target_col))

    # 전처리
    train_data = df[feature_cols + [target_col]].copy()

    for col in feature_cols + [target_col]:
        train_data[col] = pd.to_numeric(
            train_data[col],
            errors="coerce"
        )

    train_data = train_data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    st.write(f"학습에 사용할 수 있는 데이터: {len(train_data)}개")

    st.divider()

    # 모델 선택
    st.subheader("2. 머신러닝 모델 선택")

    model_type = st.radio(
        "학습할 모델",
        [
            "다중 선형회귀",
            "랜덤 포레스트",
            "XGBoost"
        ],
        horizontal=True
    )

    st.divider()

    # 학습 데이터 비율
    test_size = st.slider(
        "테스트 데이터 비율(%)",
        min_value=10,
        max_value=40,
        value=20
    )

    random_state = st.number_input(
        "랜덤 시드",
        min_value=0,
        max_value=10000,
        value=42
    )

    # 모델별 설정
    if model_type == "랜덤 포레스트":

        n_estimators = st.slider(
            "트리 개수",
            50,
            500,
            200
        )

        max_depth = st.slider(
            "최대 트리 깊이",
            2,
            20,
            8
        )

    elif model_type == "XGBoost":

        n_estimators = st.slider(
            "학습 반복 횟수",
            50,
            500,
            200
        )

        learning_rate = st.slider(
            "학습률",
            0.01,
            0.30,
            0.05,
            step=0.01
        )

        max_depth = st.slider(
            "최대 트리 깊이",
            2,
            10,
            4
        )

    st.divider()

    # 학습 실행
    if st.button(
        "🚀 모델 학습 시작",
        type="primary",
        use_container_width=True
    ):

        if len(train_data) < 10:
            st.error("학습 데이터가 부족합니다.")

        else:

            X = train_data[feature_cols]
            y = train_data[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=int(random_state)
            )

            if model_type == "다중 선형회귀":

                model = LinearRegression()

            elif model_type == "랜덤 포레스트":

                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=int(random_state),
                    n_jobs=-1
                )

            else:

                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=int(random_state),
                    objective="reg:squarederror",
                    n_jobs=-1,
                    verbosity=0
                )

            with st.spinner("모델을 학습하고 있습니다..."):

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(X_test)

            # 평가 지표
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

            # 결과 저장
            st.session_state["model_result"] = {
                "model_name": model_type,
                "target": target_col,
                "features": feature_cols,
                "model": model,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "actual": y_test.reset_index(drop=True),
                "predicted": predictions,
                "train_count": len(X_train),
                "test_count": len(X_test),
                "data_count": len(train_data)
            }

            st.success(
                "모델 학습이 완료되었습니다. "
                "학습 결과 메뉴에서 확인하세요."
            )

            st.metric("결정계수 R²", f"{r2:.4f}")


# =========================================================
# 메뉴 4. 학습 결과
# =========================================================

elif menu == "학습 결과":

    st.title("📈 학습 결과")

    st.write(
        "학습된 모델의 예측 성능과 변수 중요도를 확인합니다."
    )

    if "model_result" not in st.session_state:

        st.warning(
            "아직 학습된 모델이 없습니다. "
            "모델 학습 메뉴에서 먼저 학습을 진행하세요."
        )

    else:

        result = st.session_state["model_result"]

        model_name = result["model_name"]
        model = result["model"]

        st.subheader("1. 학습 정보")

        c1, c2, c3 = st.columns(3)

        c1.metric("사용 모델", model_name)
        c2.metric("전체 데이터", f"{result['data_count']}개")
        c3.metric("테스트 데이터", f"{result['test_count']}개")

        st.write(
            "**독립변수:** "
            + ", ".join(
                korean_name(c)
                for c in result["features"]
            )
        )

        st.write(
            "**종속변수:** "
            + korean_name(result["target"])
        )

        st.divider()

        # 성능 평가
        st.subheader("2. 모델 성능 평가")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "결정계수 R²",
            f"{result['r2']:.4f}"
        )

        c2.metric(
            "평균 절대 오차 MAE",
            f"{result['mae']:.4f}"
        )

        c3.metric(
            "평균 제곱근 오차 RMSE",
            f"{result['rmse']:.4f}"
        )

        st.caption(
            "R²는 높을수록 좋으며, MAE와 RMSE는 낮을수록 좋습니다."
        )

        st.divider()

        # 실제값 예측값 비교
        st.subheader("3. 실제값과 예측값 비교")

        comparison_df = pd.DataFrame({
            "실제값": result["actual"],
            "예측값": result["predicted"]
        })

        fig = px.scatter(
            comparison_df,
            x="실제값",
            y="예측값",
            title=f"{model_name} 실제값과 예측값"
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

        st.divider()

        # 잔차 분석
        st.subheader("4. 예측 오차 분석")

        residual_df = comparison_df.copy()

        residual_df["오차"] = (
            residual_df["실제값"]
            - residual_df["예측값"]
        )

        fig = px.histogram(
            residual_df,
            x="오차",
            nbins=30,
            title="예측 오차 분포"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # 변수 중요도
        st.subheader("5. 변수 영향력 및 중요도")

        if model_name == "다중 선형회귀":

            coefficient_df = pd.DataFrame({
                "변수": [
                    korean_name(c)
                    for c in result["features"]
                ],
                "회귀계수": model.coef_
            })

            st.dataframe(
                coefficient_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            st.write(
                f"회귀식의 절편: {model.intercept_:.4f}"
            )

            fig = px.bar(
                coefficient_df,
                x="회귀계수",
                y="변수",
                orientation="h",
                title="다중 선형회귀 회귀계수"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            importance_df = pd.DataFrame({
                "변수": [
                    korean_name(c)
                    for c in result["features"]
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
                title=f"{model_name} 변수 중요도"
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

        st.divider()

        # 결과 다운로드
        st.subheader("6. 예측 결과 다운로드")

        st.download_button(
            "📥 예측 결과 CSV 다운로드",
            data=comparison_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=f"{model_name}_결과.csv",
            mime="text/csv"
        )

        if st.button("학습 결과 초기화"):
            del st.session_state["model_result"]
            st.rerun()


# =========================================================
# 마무리
# =========================================================

st.sidebar.divider()
st.sidebar.caption("야구 타격 데이터 분석 프로젝트")
