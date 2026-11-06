import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import metrics

# Установка конфигурации страницы Streamlit
st.set_page_config(
    page_title="Wine Sommelier Assistant",
    page_icon="🍷",
    layout="wide"
)

# ---------------------------------------------------------
# Задание 4.5. Кэширование
# ---------------------------------------------------------
# Ответы на вопросы:
# 1. Почему для модели используется st.cache_resource, а для данных – st.cache_data?
#    - st.cache_resource предназначен для глобальных ресурсов и тяжелых недетерминированных
#      или неделящихся объектов (таких как объекты моделей ML, сетевые соединения, подключения к БД).
#      Модель загружается в память ровно один раз на весь сервер как singleton и разделяется
#      между всеми пользователями без копирования и сериализации.
#    - st.cache_data предназначен для возвращаемых по значению данных (DataFrames, массивы, словари).
#      Streamlit кэширует их с защитой от мутаций (mutation guard), возвращая безопасную копию.
# 2. Что изменится, если вызвать joblib.load без декоратора?
#    - Streamlit перезапускает весь Python-скрипт сверху вниз при каждом действии пользователя
#      (движение слайдера, переключение вкладок, ввод чисел). Без декоратора @st.cache_resource
#      файл model.joblib читался и десериализовался бы заново с диска при каждом клике, что
#      вызовет зависание интерфейса, лишние задержки ввода-вывода (I/O) и повышенное потребление CPU/RAM.

MODEL_PATH = Path(__file__).parent / "model.joblib"
CONFIG_PATH = Path(__file__).parent / "config.json"
TEST_PATH = Path(__file__).parent / "test.csv"


@st.cache_resource
def load_model():
    """Загрузка обученного пайплайна (StandardScaler + LogisticRegression)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели {MODEL_PATH} не найден.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    """Загрузка сохраненного оптимального порога."""
    if not CONFIG_PATH.exists():
        return {"threshold": 0.27}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_data():
    """Загрузка тестовой выборки."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Файл {TEST_PATH} не найден.")
    return pd.read_csv(TEST_PATH)


@st.cache_data
def compute_test_probabilities(_model, df_features):
    """
    Вычисление вероятностей класса 1 на тестовой выборке один раз.
    Имя аргумента _model начинается с подчеркивания, чтобы Streamlit не тратил время на ее хеширование.
    """
    return _model.predict_proba(df_features)[:, 1]

try:
    model = load_model()
    config = load_config()
    test_df = load_test_data()
    feature_cols = [c for c in test_df.columns if c != "target"]
    test_probs = compute_test_probabilities(model, test_df[feature_cols])
    y_true_test = test_df["target"].to_numpy()
except Exception as e:
    st.error(f"Ошибка при инициализации приложения: {e}")
    st.stop()

st.sidebar.title("Панель управления")
default_threshold = float(config.get("threshold", 0.27))

threshold = st.sidebar.slider(
    "Порог классификации (t)",
    min_value=0.01,
    max_value=0.99,
    value=default_threshold,
    step=0.01,
    help="Вина с вероятностью качества >= этого порога классифицируются как «хорошие»."
)

st.sidebar.markdown(f"**Текущий порог:** `{threshold:.2f}`")
st.sidebar.caption(
    f"Порог по умолчанию из `config.json`: `{default_threshold:.4f}` (оптимизирован под максимальный $F_1$ на валидации)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Памятка для сомелье:")
st.sidebar.info(
    "**Порог выше (t ↑):** Строгий отбор. Меньше плохих вин попадет к клиентам (растет точность Precision), но подборка станет меньше (падает полнота Recall).\n\n"
    "**Порог ниже (t ↓):** Широкий ассортимент. Больше хороших вин попадет в каталог (растет Recall), но возрастает риск предложить посредственное вино (падает Precision)."
)

st.title("Система классификации качества вин")
st.markdown(
    "Интерактивное рабочее место сомелье сервиса винной подписки. "
    "Модель на основе физико-химических показателей отбирает вина высшего качества для клиентов."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Оценка одного вина",
    "Модельные метрики",
    "Пакетное предсказание",
    "Что выучила модель"
])

with tab1:
    st.subheader("Лабораторный анализ образца вина")
    st.write("Введите физико-химические показатели образца. Диапазоны и значения по умолчанию взяты по медианам тестовой выборки.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.number_input("Фиксированная кислотность (fixed_acidity)", min_value=3.0, max_value=16.0, value=7.0, step=0.1)
        volatile_acidity = st.number_input("Летучая кислотность (volatile_acidity)", min_value=0.05, max_value=1.5, value=0.295, step=0.01, format="%.3f")
        citric_acid = st.number_input("Лимонная кислота (citric_acid)", min_value=0.0, max_value=2.0, value=0.30, step=0.01)
        residual_sugar = st.number_input("Остаточный сахар (residual_sugar)", min_value=0.5, max_value=30.0, value=3.10, step=0.1)

    with col2:
        chlorides = st.number_input("Хлориды (chlorides)", min_value=0.005, max_value=0.7, value=0.047, step=0.001, format="%.3f")
        free_sulfur_dioxide = st.number_input("Свободный диоксид серы (free_SO2)", min_value=1.0, max_value=150.0, value=29.0, step=1.0)
        total_sulfur_dioxide = st.number_input("Общий диоксид серы (total_SO2)", min_value=5.0, max_value=350.0, value=119.0, step=1.0)
        density = st.number_input("Плотность (density)", min_value=0.985, max_value=1.005, value=0.99493, step=0.0001, format="%.5f")

    with col3:
        pH = st.number_input("Кислотность (pH)", min_value=2.5, max_value=4.0, value=3.20, step=0.01)
        sulphates = st.number_input("Сульфаты (sulphates)", min_value=0.2, max_value=2.0, value=0.51, step=0.01)
        alcohol = st.number_input("Крепость / Алкоголь, % (alcohol)", min_value=8.0, max_value=15.0, value=10.3, step=0.1)
        wine_type = st.radio("Тип вина", ["Белое вино", "Красное вино"], horizontal=True)
        is_red = 1 if wine_type == "Красное вино" else 0

    single_wine = pd.DataFrame([{
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "is_red": is_red
    }])[feature_cols]

    proba_good = float(model.predict_proba(single_wine)[0, 1])
    is_good = proba_good >= threshold

    st.markdown("---")
    st.markdown("### Результат классификации")
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label="Вероятность высокого качества P(target=1)", value=f"{proba_good * 100:.1f}%")
        st.caption(f"Установленный порог отбора: `{threshold * 100:.1f}%`")
    
    with res_col2:
        if is_good:
            st.success("### Решение: ХОРОШЕЕ ВИНО (Рекомендовать в подборку)")
            st.write(
                f"Вероятность качества **{proba_good * 100:.1f}%** превышает порог отсечения **{threshold * 100:.1f}%**. "
                "Данный образец соответствует стандартам премиальной клиентской подборки."
            )
        else:
            st.warning("### Решение: ПОСРЕДСТВЕННОЕ ВИНО (Не включать в подборку)")
            st.write(
                f"Вероятность качества **{proba_good * 100:.1f}%** ниже порога отсечения **{threshold * 100:.1f}%**. "
                "Образец не рекомендуется для клиентов сервиса во избежание разочарования качеством."
            )

with tab2:
    st.subheader("Метрики качества модели на тестовой выборке")
    st.write(f"Текущий порог классификации: **{threshold:.2f}**. Ниже отображаются динамически пересчитанные метрики.")

    y_pred_dynamic = (test_probs >= threshold).astype(int)
    
    acc_cur = metrics.accuracy_score(y_true_test, y_pred_dynamic)
    prec_cur = metrics.precision_score(y_true_test, y_pred_dynamic, zero_division=0)
    rec_cur = metrics.recall_score(y_true_test, y_pred_dynamic, zero_division=0)
    f1_cur = metrics.f1_score(y_true_test, y_pred_dynamic, zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision (Точность)", f"{prec_cur:.4f}")
    m2.metric("Recall (Полнота)", f"{rec_cur:.4f}")
    m3.metric("F1-score", f"{f1_cur:.4f}")
    m4.metric("Accuracy", f"{acc_cur:.4f}")

    st.markdown("---")
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        st.markdown("#### Матрица ошибок (Confusion Matrix)")
        cm_cur = metrics.confusion_matrix(y_true_test, y_pred_dynamic)
        
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm_cur, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
            xticklabels=['Не хорошее', 'Хорошее'],
            yticklabels=['Не хорошее', 'Хорошее']
        )
        ax_cm.set_xlabel('Предсказание модели')
        ax_cm.set_ylabel('Истинный класс')
        ax_cm.set_title(f'Матрица ошибок (порог t = {threshold:.2f})')
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with plot_col2:
        st.markdown("#### ROC- и PR-кривые с текущим порогом")
        fig_curves, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 3.8))

        fpr, tpr, _ = metrics.roc_curve(y_true_test, test_probs)
        roc_auc = metrics.roc_auc_score(y_true_test, test_probs)
        ax_roc.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', color='royalblue', lw=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)

        tn, fp, fn, tp = cm_cur.ravel()
        cur_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        cur_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax_roc.scatter([cur_fpr], [cur_tpr], color='red', s=80, zorder=5, label=f't = {threshold:.2f}')
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR (Recall)')
        ax_roc.set_title('ROC-кривая')
        ax_roc.legend(fontsize=8)
        ax_roc.grid(True)

        prec_arr, rec_arr, _ = metrics.precision_recall_curve(y_true_test, test_probs)
        pr_auc = metrics.average_precision_score(y_true_test, test_probs)
        ax_pr.plot(rec_arr, prec_arr, label=f'PR (AUC = {pr_auc:.3f})', color='forestgreen', lw=2)
        ax_pr.axhline(y=y_true_test.mean(), color='crimson', linestyle=':', label='Baseline')

        ax_pr.scatter([rec_cur], [prec_cur], color='red', s=80, zorder=5, label=f't = {threshold:.2f}')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('PR-кривая')
        ax_pr.legend(fontsize=8)
        ax_pr.grid(True)

        plt.tight_layout()
        st.pyplot(fig_curves)
        plt.close(fig_curves)

    st.info(
        "💡 **Интерпретация для сомелье:**\n\n"
        f"- При текущем пороге **{threshold:.2f}** модель находит **{rec_cur * 100:.1f}%** всех существующих хороших вин (Recall), "
        f"при этом среди вин, рекомендованных моделью, **{prec_cur * 100:.1f}%** действительно являются отличными (Precision).\n"
        f"- Если вы **увеличите порог**, отбор станет строже: Precision вырастет (меньше посредственных вин попадет клиентам), но Recall упадет (многие хорошие вина будут отсеяны).\n"
        f"- Если вы **уменьшите порог**, отбор станет мягче: Recall вырастет (в подборку попадет больше разнообразных вин), но Precision снизится (возрастет риск прислать подписчику посредственное вино)."
    )

with tab3:
    st.subheader("Пакетная оценка партии вин из CSV-файла")
    st.write(
        "Загрузите CSV-файл с лабораторными анализами образцов вин (12 признаков без столбца `target`). "
        "Модель рассчитает вероятности и сформирует рекомендацию для каждого вина при текущем пороге."
    )
    sample_df = test_df[feature_cols].head(5)
    csv_sample = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать пример CSV-файла для тестирования",
        data=csv_sample,
        file_name="sample_wines_batch.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Загрузите CSV-файл с признаками вин", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Загружено строк: `{batch_df.shape[0]}`, столбцов: `{batch_df.shape[1]}`")
            missing_columns = [col for col in feature_cols if col not in batch_df.columns]
            if missing_columns:
                st.error(
                    f"Ошибка в структуре файла! В загруженном датасете отсутствуют обязательные столбцы:\n\n"
                    f"`{missing_columns}`\n\n"
                    f"Пожалуйста, убедитесь, что файл содержит все 12 признаков: `{feature_cols}`."
                )
            else:
                input_features = batch_df[feature_cols]
                batch_probs = model.predict_proba(input_features)[:, 1]
                batch_decisions = (batch_probs >= threshold).astype(int)

                result_df = batch_df.copy()
                result_df["Вероятность качества"] = np.round(batch_probs, 4)
                result_df["Решение модели"] = np.where(batch_decisions == 1, "Хорошее вино (в подборку)", "Посредственное")

                good_count = int(np.sum(batch_decisions == 1))
                total_count = len(batch_decisions)

                st.success(
                    f"Пакетная классификация успешно завершена! "
                    f"Отобрано хороших вин: **{good_count}** из **{total_count}** ({good_count / total_count * 100:.1f}%)."
                )

                st.dataframe(result_df, use_container_width=True)

                csv_result = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты с предсказаниями (CSV)",
                    data=csv_result,
                    file_name="wine_predictions_result.csv",
                    mime="text/csv"
                )

        except Exception as err:
            st.error(f"Не удалось обработать файл: {err}")

with tab4:
    st.subheader("Интерпретация коэффициентов логистической регрессии")
    st.write(
        "Ниже показаны коэффициенты модели для каждого признака после масштабирования (`StandardScaler`). "
        "Поскольку признаки нормализованы, абсолютная величина коэффициента отражает силу влияния фактора на качество вина."
    )

    try:
        logistic_step = model.named_steps['logisticregression']
        coefs = logistic_step.coef_[0]
        
        coef_df = pd.DataFrame({
            "Признак": feature_cols,
            "Вес (коэффициент)": coefs
        }).sort_values(by="Вес (коэффициент)", ascending=True)

        fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
        colors = ['crimson' if c < 0 else 'forestgreen' for c in coef_df["Вес (коэффициент)"]]
        bars = ax_coef.barh(coef_df["Признак"], coef_df["Вес (коэффициент)"], color=colors, alpha=0.85, edgecolor='black')

        ax_coef.axvline(0, color='black', lw=1.2, linestyle='--')
        ax_coef.set_xlabel("Значение стандартизированного коэффициента", fontsize=11)
        ax_coef.set_title("Влияние признаков на вероятность класса «Хорошее вино»", fontsize=13)
        ax_coef.grid(True, axis='x', linestyle=':', alpha=0.6)

        for bar in bars:
            val = bar.get_width()
            offset = 0.03 if val >= 0 else -0.12
            ax_coef.text(val + offset, bar.get_y() + bar.get_height() / 4, f"{val:+.2f}", fontsize=9, fontweight='bold')

        st.pyplot(fig_coef)
        plt.close(fig_coef)

        st.markdown("### Анализ выводов модели:")
        st.markdown(
            "- **Сильнее всего голосуют ЗА хорошее вино (положительные веса):**\n"
            "  1. `residual_sugar` (+1.00) — более выраженная полнотелость и гармоничный баланс сахаров.\n"
            "  2. `fixed_acidity` (+0.60) — хорошая базовая структура кислоты вина.\n"
            "  3. `alcohol` (+0.60) — более зрелый виноград и высокая спиртуозность.\n"
            "  4. `pH` (+0.38) и `sulphates` (+0.35) — стабильность и антиоксидантная защита вина.\n\n"
            "- **Сильнее всего голосуют ПРОТИВ хорошего вина (отрицательные веса):**\n"
            "  1. `density` (-1.17) — избыточная водянистость/плотность снижает органолептическую оценку.\n"
            "  2. `volatile_acidity` (-0.60) — летучая уксусная кислотность (признак уксусного скисания и дефекта вина).\n"
            "  3. `total_sulfur_dioxide` (-0.25) и `chlorides` (-0.19) — избыток диоксида серы (химический привкус) и минеральных солей (солоноватый привкус)."
        )

    except Exception as e:
        st.error(f"Не удалось извлечь веса модели: {e}")
