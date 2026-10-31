
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


# -----------------------------------
# 기본 설정
# -----------------------------------

st.set_page_config(
    page_title="야구 데이터 분석실",
    page_icon="⚾",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/leeseojin09/baseball/refs/heads/main/data/exit_velocity.csv"

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


def percent_value(value):
    """0~1 비율이면 100을 곱하고, 0~100이면 그대로 표시."""
    if pd.isna(value):
        return "자료 없음"

    if abs(value) <= 1:
        value *= 100

    return f"{value:.2f}%"


# -----------------------------------
# 데이터 불러오기
# -----------------------------------

@st.cache_data
def load_data():
    response = requests.get(DATA_URL, timeout=20)
    response.raise_for_status()

    return pd.read_csv(BytesIO(response.content))


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# -----------------------------------
# 제목 및 사이드바
# -----------------------------------

st.title("⚾ 야구 데이터 분석실")
st.caption("타구 데이터를 활용한 배럴 타구 분석 및 머신러닝")

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "데이터 탐색",
        "데이터 전처리",
        "모델 학습",
        "학습 결과"
    ]
)


# -----------------------------------
# 데이터 탐색
# -----------------------------------

if menu == "데이터 탐색":

    st.header("1. 데이터 탐색")

    st.subheader("전체 데이터")

    display_df = df.copy()

    if NAME_COL in display_df.columns:
        display_df[NAME_COL] = (
            display_df[NAME_COL]
            .fillna("")
            .astype(str)
            .str.replace(",", " ", regex=False)
            .str.strip()
        )

    display_df = display_df.rename(
        columns={col: korean_name(col) for col in display_df.columns}
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

    st.download_button(
        "전체 데이터 CSV 다운로드",
        df.to_csv(index=False).encode("utf-8-sig"),
        "baseball_data.csv",
        "text/csv"
    )

    st.divider()

    # 주요 지표 평균
    st.subheader("2. 주요 타격 지표 평균")

    metric_cols = st.columns(5)

    for i, col in enumerate(MAIN_COLS):

        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            value = values.mean()

            with metric_cols[i]:
                if col in ["brl_percent", "fbld", "ev95percent"]:
                    st.metric(
                        korean_name(col),
                        percent_value(value)
                    )
                else:
                    st.metric(
                        korean_name(col),
                        f"{value:.2f}" if pd.notna(value) else "자료 없음"
                    )

    # 배럴 타구 평균 추가
    st.subheader("⚾ 배럴 타구 평균")

    barrel_cols = st.columns(2)

    if "barrels" in df.columns:

        barrel_mean = pd.to_numeric(
            df["barrels"],
            errors="coerce"
        ).mean()

        with barrel_cols[0]:
            st.metric(
                "평균 배럴 타구 수",
                f"{barrel_mean:.2f}" if pd.notna(barrel_mean) else "자료 없음"
            )

    if "brl_percent" in df.columns:

        barrel_percent_mean = pd.to_numeric(
            df["brl_percent"],
            errors="coerce"
        ).mean()

        with barrel_cols[1]:
            st.metric(
                "평균 배럴 타구 비율",
                percent_value(barrel_percent_mean)
            )

    st.caption(
        "배럴 타구 수와 배럴 타구 비율은 전체 선수 데이터의 산술평균입니다."
    )

    st.divider()

    # 기초 통계
    st.subheader("3. 기초 통계량")

    numeric_df = df.copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(
            numeric_df[col],
            errors="coerce"
        )

    numeric_df = numeric_df.select_dtypes(include=np.number)

    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )

    st.divider()

    # 히스토그램
    st.subheader("4. 변수별 분포")

    available_cols = [
        col for col in MAIN_COLS
        if col in numeric_df.columns
    ]

    selected_hist = st.selectbox(
        "분포를 확인할 변수",
        available_cols,
        format_func=korean_name
    )

    fig = px.histogram(
        df,
        x=selected_hist,
        nbins=30,
        title=f"{korean_name(selected_hist)} 분포"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 선수별 순위
    st.subheader("5. 선수별 순위")

    rank_cols = [
        col for col in [
            "brl_percent",
            "barrels",
            "avg_hit_angle",
            "avg_distance",
            "avg_hit_speed"
        ]
        if col in df.columns
    ]

    rank_col = st.selectbox(
        "순위를 확인할 지표",
        rank_cols,
        format_func=korean_name
    )

    top_n = st.slider(
        "표시할 선수 수",
        5, 30, 10
    )

    ranking = df.copy()
    ranking[rank_col] = pd.to_numeric(
        ranking[rank_col],
        errors="coerce"
    )

    ranking = ranking.sort_values(
        rank_col,
        ascending=False
    ).head(top_n)

    if NAME_COL in ranking.columns:
        ranking[NAME_COL] = (
            ranking[NAME_COL]
            .fillna("")
            .astype(str)
            .str.replace(",", " ", regex=False)
            .str.strip()
        )

    st.dataframe(
        ranking[
            [NAME_COL, rank_col]
            if NAME_COL in ranking.columns
            else [rank_col]
        ].rename(
            columns={col: korean_name(col) for col in ranking.columns}
        ),
        use_container_width=True
    )

    st.divider()

    # 상관관계 분석
    st.subheader("6. 전체 수치형 변수 상관관계")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="전체 변수 상관관계 히트맵"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("7. 주요 변수 상관관계")

    selected_main = [
        col for col in MAIN_COLS
        if col in numeric_df.columns
    ]

    main_corr = numeric_df[selected_main].corr()

    fig = px.imshow(
        main_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="주요 타격 지표 상관관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 산점도와 회귀선
    st.subheader("8. 변수 간 관계 분석")

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox(
            "독립변수",
            selected_main,
            index=1 if len(selected_main) > 1 else 0,
            format_func=korean_name
        )

    with col2:
        y_col = st.selectbox(
            "종속변수",
            selected_main,
            index=0,
            format_func=korean_name
        )

    plot_df = df[[x_col, y_col]].copy()

    plot_df[x_col] = pd.to_numeric(
        plot_df[x_col],
        errors="coerce"
    )

    plot_df[y_col] = pd.to_numeric(
        plot_df[y_col],
        errors="coerce"
    )

    plot_df = plot_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(plot_df) >= 2:

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            trendline="ols",
            title=f"{korean_name(x_col)}와 {korean_name(y_col)}의 관계"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        correlation = plot_df[x_col].corr(plot_df[y_col])

        st.metric(
            "피어슨 상관계수",
            f"{correlation:.3f}"
        )

    else:
        st.warning("분석할 데이터가 부족합니다.")


# -----------------------------------
# 데이터 전처리
# -----------------------------------

elif menu == "데이터 전처리":

    st.header("2. 데이터 전처리")

    st.subheader("데이터 자료형")

    type_df = pd.DataFrame({
        "변수": df.columns,
        "자료형": df.dtypes.astype(str).values,
        "결측값 수": df.isnull().sum().values
    })

    st.dataframe(
        type_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("분석 변수 선택")

    available_main = [
        col for col in MAIN_COLS
        if col in df.columns
    ]

    target = st.selectbox(
        "종속변수",
        available_main,
        format_func=korean_name
    )

    feature_options = [
        col for col in available_main
        if col != target
    ]

    features = st.multiselect(
        "독립변수",
        feature_options,
        default=feature_options,
        format_func=korean_name
    )

    if st.button("데이터 전처리 실행"):

        if len(features) == 0:
            st.error("독립변수를 하나 이상 선택하세요.")

        else:

            selected_cols = [target] + features

            clean_df = df[selected_cols].copy()

            for col in selected_cols:
                clean_df[col] = pd.to_numeric(
                    clean_df[col],
                    errors="coerce"
                )

            clean_df = clean_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            before = len(clean_df)

            clean_df = clean_df.dropna()

            after = len(clean_df)

            st.session_state["clean_df"] = clean_df
            st.session_state["target"] = target
            st.session_state["features"] = features

            st.success("전처리가 완료되었습니다.")

            c1, c2, c3 = st.columns(3)

            c1.metric("전처리 전 데이터", f"{before}개")
            c2.metric("전처리 후 데이터", f"{after}개")
            c3.metric("제거된 데이터", f"{before - after}개")

            st.subheader("전처리 결과")

            st.dataframe(
                clean_df.rename(
                    columns={
                        col: korean_name(col)
                        for col in clean_df.columns
                    }
                ),
                use_container_width=True
            )

            st.download_button(
                "전처리 데이터 다운로드",
                clean_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                "clean_baseball_data.csv",
                "text/csv"
            )


# -----------------------------------
# 모델 학습
# -----------------------------------

elif menu == "모델 학습":

    st.header("3. 머신러닝 모델 학습")

    st.info(
        "다중 선형회귀, 랜덤 포레스트, XGBoost 모델을 이용하여 "
        "선수의 타격 지표를 예측합니다."
    )

    available_main = [
        col for col in MAIN_COLS
        if col in df.columns
    ]

    target = st.selectbox(
        "예측할 종속변수",
        available_main,
        format_func=korean_name
    )

    feature_options = [
        col for col in available_main
        if col != target
    ]

    features = st.multiselect(
        "학습에 사용할 독립변수",
        feature_options,
        default=feature_options,
        format_func=korean_name
    )

    model_name = st.radio(
        "학습 모델 선택",
        [
            "다중 선형회귀",
            "랜덤 포레스트",
            "XGBoost"
        ],
        horizontal=True
    )

    st.divider()

    test_size = st.slider(
        "테스트 데이터 비율",
        0.1, 0.4, 0.2, 0.05
    )

    random_state = st.number_input(
        "랜덤 시드",
        min_value=0,
        value=42
    )

    st.subheader("모델 설정")

    if model_name == "랜덤 포레스트":

        n_estimators = st.slider(
            "트리 개수",
            50, 500, 100, 50
        )

        max_depth = st.slider(
            "최대 깊이",
            2, 20, 5
        )

    elif model_name == "XGBoost":

        n_estimators = st.slider(
            "반복 학습 횟수",
            50, 500, 100, 50
        )

        max_depth = st.slider(
            "트리 최대 깊이",
            2, 10, 3
        )

        learning_rate = st.slider(
            "학습률",
            0.01, 0.3, 0.1, 0.01
        )

    if st.button("모델 학습 시작", type="primary"):

        if len(features) == 0:
            st.error("독립변수를 하나 이상 선택하세요.")

        else:

            selected_cols = [target] + features

            model_df = df[selected_cols].copy()

            for col in selected_cols:
                model_df[col] = pd.to_numeric(
                    model_df[col],
                    errors="coerce"
                )

            model_df = model_df.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(model_df) < 5:
                st.error("학습에 사용할 데이터가 부족합니다.")

            else:

                X = model_df[features]
                y = model_df[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=int(random_state)
                )

                if model_name == "다중 선형회귀":

                    model = LinearRegression()

                elif model_name == "랜덤 포레스트":

                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=int(random_state),
                        n_jobs=-1
                    )

                else:

                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=int(random_state),
                        objective="reg:squarederror",
                        n_jobs=-1
                    )

                with st.spinner("모델을 학습하고 있습니다..."):

                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)

                r2 = r2_score(
                    y_test,
                    predictions
                )

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

                result_df = pd.DataFrame({
                    "실제값": y_test.values,
                    "예측값": predictions,
                    "잔차": y_test.values - predictions
                })

                st.session_state["model_result"] = {
                    "model": model,
                    "model_name": model_name,
                    "target": target,
                    "features": features,
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "result_df": result_df,
                    "y_test": y_test,
                    "predictions": predictions
                }

                st.success("모델 학습이 완료되었습니다!")

                c1, c2, c3 = st.columns(3)

                c1.metric("R² 결정계수", f"{r2:.4f}")
                c2.metric("MAE 평균 절대 오차", f"{mae:.4f}")
                c3.metric("RMSE 평균 제곱근 오차", f"{rmse:.4f}")

                st.info(
                    "자세한 분석 결과는 사이드바의 '학습 결과' 메뉴에서 확인하세요."
                )


# -----------------------------------
# 학습 결과
# -----------------------------------

elif menu == "학습 결과":

    st.header("4. 모델 학습 결과")

    if "model_result" not in st.session_state:

        st.warning(
            "아직 학습 결과가 없습니다. 먼저 '모델 학습' 메뉴에서 모델을 학습하세요."
        )

    else:

        result = st.session_state["model_result"]

        model = result["model"]
        model_name = result["model_name"]
        target = result["target"]
        features = result["features"]

        st.subheader("학습 정보")

        c1, c2, c3 = st.columns(3)

        c1.metric("사용 모델", model_name)
        c2.metric("종속변수", korean_name(target))
        c3.metric("독립변수 수", len(features))

        st.divider()

        st.subheader("1. 모델 성능 평가")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "R² 결정계수",
            f"{result['r2']:.4f}"
        )

        c2.metric(
            "MAE 평균 절대 오차",
            f"{result['mae']:.4f}"
        )

        c3.metric(
            "RMSE 평균 제곱근 오차",
            f"{result['rmse']:.4f}"
        )

        st.caption(
            "R²는 설명력을, MAE와 RMSE는 예측 오차의 크기를 나타냅니다. "
            "MAE와 RMSE는 낮을수록 좋습니다."
        )

        st.divider()

        # 실제값과 예측값
        st.subheader("2. 실제값과 예측값 비교")

        result_df = result["result_df"]

        fig = px.scatter(
            result_df,
            x="실제값",
            y="예측값",
            title="실제값과 예측값 비교"
        )

        min_value = min(
            result_df["실제값"].min(),
            result_df["예측값"].min()
        )

        max_value = max(
            result_df["실제값"].max(),
            result_df["예측값"].max()
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
        st.subheader("3. 잔차 분포")

        fig = px.histogram(
            result_df,
            x="잔차",
            nbins=30,
            title="잔차 히스토그램"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # 계수 및 변수 중요도
        st.subheader("4. 변수별 영향력 분석")

        if model_name == "다중 선형회귀":

            importance_df = pd.DataFrame({
                "변수": features,
                "회귀계수": model.coef_
            })

            importance_df["변수"] = importance_df["변수"].map(
                korean_name
            )

            fig = px.bar(
                importance_df,
                x="변수",
                y="회귀계수",
                title="다중 선형회귀 변수별 회귀계수"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            st.caption(
                "회귀계수는 다른 변수가 일정할 때 해당 변수가 한 단위 증가하면 "
                "예측값이 얼마나 변하는지를 나타냅니다."
            )

        else:

            importance_df = pd.DataFrame({
                "변수": features,
                "중요도": model.feature_importances_
            })

            importance_df["변수"] = importance_df["변수"].map(
                korean_name
            )

            importance_df = importance_df.sort_values(
                "중요도",
                ascending=False
            )

            fig = px.bar(
                importance_df,
                x="변수",
                y="중요도",
                title=f"{model_name} 변수 중요도"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            st.caption(
                "변수 중요도는 모델이 예측 과정에서 각 변수를 얼마나 활용했는지 나타냅니다. "
                "인과관계를 의미하지는 않습니다."
            )

        st.divider()

        # 예측 결과 다운로드
        st.subheader("5. 예측 결과 다운로드")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.download_button(
            "예측 결과 CSV 다운로드",
            result_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            "baseball_prediction_results.csv",
            "text/csv"
        )

        if st.button("학습 결과 초기화"):

            del st.session_state["model_result"]

            st.rerun()
