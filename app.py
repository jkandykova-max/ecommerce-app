import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.sparse import hstack

# ===============================
# 1. ЗАГРУЗКА МОДЕЛИ И АРТЕФАКТОВ
# ===============================

@st.cache_resource
def load_model_and_tools():
    try:
        model = joblib.load("best_model.pkl")
        tfidf = joblib.load("tfidf.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError as e:
        st.error(f"Файл {e.filename} не найден. "
                 "Убедитесь, что best_model.pkl, tfidf.pkl и scaler.pkl лежат рядом с app.py в репозитории.")
        st.stop()
    return model, tfidf, scaler

model, tfidf, scaler = load_model_and_tools()

# ПОДСТАВЬ СВОИ МЕТРИКИ!
BEST_MODEL_NAME = "Random Forest"      
BEST_ACC = 0.9107763615295481                        
BEST_F1 = 0.9111880046136102                         

# ===============================
# 2. ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
# ===============================

@st.cache_data
def load_data_from_file(uploaded_file):
    # В облаке работаем ТОЛЬКО с файлом, загруженным пользователем
    df = pd.read_csv(uploaded_file, encoding="latin1")
    return df

def prepare_product_level_df(df: pd.DataFrame):
    df = df.copy()
    df = df[df["Quantity"] > 0]
    df = df.dropna(subset=["Description"])
    df["Description_clean"] = df["Description"].str.lower().str.strip()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    product_df = df.groupby(["StockCode", "Description_clean"], as_index=False).agg({
        "Quantity": "sum",
        "UnitPrice": "mean",
        "InvoiceNo": "count",
        "CustomerID": pd.Series.nunique
    })

    product_df.rename(columns={
        "Quantity": "TotalSales",
        "UnitPrice": "AvgPrice",
        "InvoiceNo": "OrderCount",
        "CustomerID": "UniqueCustomers"
    }, inplace=True)

    median_sales = product_df["TotalSales"].median()
    product_df["Success"] = (product_df["TotalSales"] > median_sales).astype(int)

    return df, product_df

def plot_wordcloud(product_df: pd.DataFrame):
    all_text = " ".join(product_df["Description_clean"].tolist())
    wc = WordCloud(width=1600, height=800, background_color="white").generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ===============================
# 3. НАЧАЛО ПРИЛОЖЕНИЯ STREAMLIT
# ===============================

st.set_page_config(page_title="ProductSuccess Predictor", layout="wide")

st.title("ProductSuccess Predictor")
st.markdown("""
**Цель стартапа:**  
Предсказать успешность нового товара по его описанию, цене и ожидаемому спросу.
""")

tab_eda, tab_model = st.tabs(["📊 Data Overview & EDA", "🤖 Predictive Model"])

# ===============================
# TAB 1 — DATA OVERVIEW & EDA
# ===============================

with tab_eda:
    st.subheader("Разведочный анализ данных (EDA)")

    st.markdown("Загрузите CSV-файл с данными (например, Ecommerce Data) для просмотра EDA.")
    uploaded_file = st.file_uploader("Загрузите CSV с данными", type=["csv"])

    if uploaded_file is None:
        st.info("Файл ещё не загружен. Загрузите CSV, чтобы увидеть анализ.")
    else:
        df = load_data_from_file(uploaded_file)

        st.write("Первые строки набора данных:")
        st.dataframe(df.head())

        st.write("Размер датасета:", df.shape)

        df_tx, product_df = prepare_product_level_df(df)

        # 1. TOP-20 товаров по продажам
        st.markdown("### 1. TOP-20 товаров по продажам")
        top20_sales = product_df.sort_values("TotalSales", ascending=False).head(20)
        fig1 = px.bar(
            top20_sales,
            x="TotalSales",
            y="Description_clean",
            orientation="h",
            labels={"TotalSales": "Общее количество продаж", "Description_clean": "Товар"},
            height=600
        )
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, use_container_width=True)

        # 2. TOP-10 стран по количеству покупок
        if "Country" in df_tx.columns:
            st.markdown("### 2. TOP-10 стран по количеству покупок")
            country_sales = df_tx.groupby("Country")["Quantity"].sum().sort_values(ascending=False)
            top_countries = country_sales.head(10).reset_index()
            fig2 = px.bar(
                top_countries,
                x="Country",
                y="Quantity",
                labels={"Country": "Страна", "Quantity": "Количество покупок"},
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 3. Распределение средних цен по товарам
        st.markdown("### 3. Распределение средних цен по товарам")
        fig3 = px.histogram(
            product_df,
            x="AvgPrice",
            nbins=50,
            labels={"AvgPrice": "Средняя цена товара"},
            height=400
        )
        fig3.update_xaxes(range=[0, product_df["AvgPrice"].quantile(0.99)])
        st.plotly_chart(fig3, use_container_width=True)

        # 4. Динамика продаж по месяцам
        if "InvoiceDate" in df_tx.columns:
            st.markdown("### 4. Динамика продаж по месяцам")
            df_tx["InvoiceDate"] = pd.to_datetime(df_tx["InvoiceDate"])
            df_tx["Month"] = df_tx["InvoiceDate"].dt.to_period("M")
            monthly_sales = df_tx.groupby("Month")["Quantity"].sum().reset_index()
            monthly_sales["Month"] = monthly_sales["Month"].astype(str)

            fig4 = px.line(
                monthly_sales,
                x="Month",
                y="Quantity",
                labels={"Month": "Месяц", "Quantity": "Количество продаж"},
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)

        # 5. WordCloud
        st.markdown("### 5. Облако слов по названиям товаров")
        fig_wc = plot_wordcloud(product_df)
        st.pyplot(fig_wc)

# ===============================
# TAB 2 — PREDICTIVE MODEL
# ===============================

with tab_model:
    st.subheader("Предсказание успешности товара")

    st.markdown("#### Качество лучшей модели:")
    st.write(f"**Модель:** {BEST_MODEL_NAME}")
    st.write(f"**Accuracy:** {BEST_ACC:.3f}")
    st.write(f"**F1-score:** {BEST_F1:.3f}")
   

    st.markdown("----")
    st.markdown("### Сделать прогноз")

    with st.form("prediction_form"):
        description_input = st.text_area("Описание товара")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_price_input = st.number_input("Цена товара", min_value=0.0, value=10.0)

        with col2:
            order_count_input = st.number_input("Ожидаемое количество заказов", min_value=0.0, value=5.0)

        with col3:
            unique_customers_input = st.number_input("Ожидаемое число уникальных покупателей", min_value=0.0, value=3.0)

        submit_btn = st.form_submit_button("Предсказать")

    if submit_btn:
        if not description_input.strip():
            st.error("Введите описание товара.")
        else:
            X_text = tfidf.transform([description_input.lower().strip()])
            X_num = np.array([[avg_price_input, order_count_input, unique_customers_input]])
            X_num_scaled = scaler.transform(X_num)
            X_input = hstack([X_text, X_num_scaled])

            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]

            if pred == 1:
                st.success(f"Товар **БУДЕТ УСПЕШНЫМ** 🎉 (вероятность: {proba:.2%})")
            else:
                st.warning(f"Товар **может быть неуспешен** 😕 (вероятность: {proba:.2%})")
