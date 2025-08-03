# import pandas as pd
# import numpy as np
# import logging

# # ----------------------------
# # 🔧 Настройка логирования
# # ----------------------------
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# handler = logging.StreamHandler()
# formatter = logging.Formatter("[%(levelname)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# # ----------------------------
# # 🧩 Заполнение пропущенных значений
# # ----------------------------
# def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     cat_cols = df.select_dtypes(include=["object", "category"]).columns
#     df[cat_cols] = df[cat_cols].fillna("Unknown")
#     logger.info(f"Заполнены пропущенные значения в категориальных колонках: {list(cat_cols)}")
#     return df

# # ----------------------------
# # 🚫 Удаление выбросов
# # ----------------------------
# def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
#     df = df.copy()
#     initial_len = len(df)

#     for col in columns:
#         if col not in df.columns:
#             logger.warning(f"Столбец '{col}' не найден. Пропускаем.")
#             continue

#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         low = Q1 - factor * IQR
#         high = Q3 + factor * IQR
#         before = len(df)
#         df = df[(df[col] >= low) & (df[col] <= high)]
#         after = len(df)
#         removed = before - after
#         logger.info(f"Удалено {removed} выбросов по колонке '{col}'")

#     logger.info(f"Итого удалено {initial_len - len(df)} строк с выбросами")
#     return df

# # ----------------------------
# # 🧼 Удаление ненужных признаков
# # ----------------------------
# def drop_unused_columns(df: pd.DataFrame, keep_columns: list[str]) -> pd.DataFrame:
#     all_cols = set(df.columns)
#     keep_cols = set(keep_columns)
#     drop_cols = list(all_cols - keep_cols)

#     logger.info(f"Удаляем {len(drop_cols)} неиспользуемых признаков: {drop_cols}")
#     return df[list(keep_cols)]

# # ----------------------------
# # 🧼 Главный пайплайн
# # ----------------------------
# def run_pipeline(df: pd.DataFrame, outlier_columns: list[str] = None, final_features: list[str] = None) -> pd.DataFrame:
#     logger.info("🔄 Запуск пайплайна предобработки...")
#     df = fill_missing_values(df)

#     if outlier_columns:
#         df = remove_outliers_iqr(df, outlier_columns)

#     if final_features:
#         df = drop_unused_columns(df, final_features)

#     logger.info(f"✅ Готово. Размер финального датасета: {df.shape}")
#     return df



import pandas as pd
import numpy as np
from statsmodels.stats.stattools import medcouple
import logging
# ----------------------------
# 🔧 Настройка логирования
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ----------------------------
#  Заполнение пропущенных значений
# ----------------------------
def fill_missing_values(df: pd.DataFrame, num_strategy: str = None) -> pd.DataFrame:

    """
    Заполняет пропущенные значения в DataFrame согласно заданным стратегиям.
    
    Параметры:
        df: Исходный DataFrame
        num_strategy: Стратегия обработки числовых пропусков:
            - 'median': заполнение медианой
            - 'mean': заполнение средним
            - 'flag': добавление флага пропуска + заполнение медианой
            - 'delete': удаление строк с пропусками
            - None: пропуски не заполняются (по умолчанию)
            
    Возвращает:
        Обработанный DataFrame
    """


    # Категориальные признаки 
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    logger.info(f"Заполнены пропущенные значения в категориальных колонках: {list(cat_cols)}")

    # Числовые признаки
    num_cols = df.select_dtypes(include=["number"]).columns
    if num_strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif num_strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif num_strategy == "flag":
        for col in num_cols:
            df[f"{col}_is_na"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())
    elif num_strategy == 'delete':
            initial_rows = len(df)
            df = df.dropna(subset=num_cols)
            logger.info(f"Удалено {initial_rows - len(df)} строк с пропусками в числовых колонках")
            return df
    else:
        df[num_cols] = df[num_cols].fillna(np.nan)
    
    logger.info(f"Заполнены пропуски: категориальные -> 'Unknown', числовые -> {num_strategy}")
    return df




# Удаление выбросов с помощью IQR
# ----------------------------
def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    
    """
    Удаляет выбросы используя метод IQR, который подходит для  распределения близкого к нормальному.
    
    Параметры:
        df: DataFrame для обработки
        columns: Список числовых колонок для анализа
        factor: Пороговое значение для определения выбросов (обычно 3.5)
    
    Возвращает:
        DataFrame с удаленными выбросами
    """


    initial_len = len(df)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Столбец '{col}' не найден. Пропускаем.")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - factor * IQR
        high = Q3 + factor * IQR
        before = len(df)
        df = df[(df[col] >= low) & (df[col] <= high)]
        after = len(df)
        removed = before - after
        logger.info(f"Удалено {removed} выбросов по колонке '{col}'")

    logger.info(f"Итого удалено {initial_len - len(df)} строк с выбросами")
    return df




#  Удаление выбросов через medcouple (более устойчивый метод для асимметричных распределений)



def remove_outliers_medcouple(df: pd.DataFrame, columns: list[str], threshold: float = 3.5) -> pd.DataFrame:
    """
    Удаляет выбросы используя метод medcouple, который лучше работает с асимметричными распределениями.
    
    Параметры:
        df: DataFrame для обработки
        columns: Список числовых колонок для анализа
        threshold: Пороговое значение для определения выбросов (обычно 3.5)
    
    Возвращает:
        DataFrame с удаленными выбросами
    """

    initial_len = len(df)
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Столбец '{col}' не найден. Пропускаем.")
            continue
            
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        # Вычисляем medcouple
        mc = medcouple(data.values)
        
        # Вычисляем квантили
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Определяем границы в зависимости от направления асимметрии
        if mc >= 0:
            lower_bound = Q1 - threshold * np.exp(-3.5 * mc) * IQR
            upper_bound = Q3 + threshold * np.exp(4 * mc) * IQR
        else:
            lower_bound = Q1 - threshold * np.exp(-4 * mc) * IQR
            upper_bound = Q3 + threshold * np.exp(3.5 * mc) * IQR
            
        before = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        after = len(df)
        removed = before - after
        logger.info(f"Удалено {removed} выбросов по колонке '{col}' (medcouple={mc:.2f})")
    
    logger.info(f"Итого удалено {initial_len - len(df)} строк с выбросами (метод medcouple)")
    return df




#  Удаление ненужных признаков

def drop_unused_columns(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    """
    Удаляет указанные столбцы из DataFrame и возвращает копию с оставшимися столбцами.
    
    Параметры:
        df : pd.DataFrame
            Исходный DataFrame для обработки
        drop_cols : list[str]
            Список названий столбцов для удаления. Если переданные имена отсутствуют в DataFrame,
            функция проигнорирует их.
            
    Возвращает:
        pd.DataFrame
            Новый DataFrame без указанных столбцов    
    """
    all_cols = set(df.columns)
    drop_cols = set(drop_cols)
    keep_cols = list(all_cols - drop_cols)

    logger.info(f"Удаляем {len(drop_cols)} неиспользуемых признаков: {drop_cols}")
    return df[list(keep_cols)]




#  Главный пайплайн



def run_pipeline(df: pd.DataFrame, outlier_columns_iqr: list[str] = None, outlier_columns_medcouple: list[str] = None, columns_to_delete: list[str] = None) -> pd.DataFrame:
    """
    Основной пайплайн предобработки данных, выполняющий последовательно:
    1. Заполнение пропущенных значений
    2. Удаление указанных столбцов
    3. Удаление выбросов (одним или несколькими методами)
    
    Параметры:
        df : pd.DataFrame
            Исходный DataFrame для обработки
        outlier_columns_iqr : list[str], optional
            Список колонок для обработки методом IQR (для нормальных распределений)
        outlier_columns_medcouple : list[str], optional
            Список колонок для обработки методом MedCouple (для асимметричных распределений)
        columns_to_delete : list[str], optional
            Список колонок для полного удаления из набора данных
            
    Возвращает:
        pd.DataFrame
            Очищенный DataFrame после всех преобразований
    """
    
    logger.info("🔄 Запуск пайплайна предобработки...")
    df = df.copy()
    #Заполнение пропущенных значений
    df = fill_missing_values(df)


    #Удаление ненужных колонок:
    if columns_to_delete:
        df = drop_unused_columns(df, columns_to_delete)

    #Удаление выбросов

    if outlier_columns_iqr and outlier_columns_medcouple:
        common_cols = set(outlier_columns_iqr) & set(outlier_columns_medcouple)
        if common_cols:
            logger.warning(f"Колонки {common_cols} обрабатываются обоими методами")
            df1 = remove_outliers_medcouple(df, outlier_columns_medcouple)
            df2 = remove_outliers_iqr(df, outlier_columns_iqr)
            df = df.loc[df1.index.intersection(df2.index)]   
    elif outlier_columns_iqr:
        df = remove_outliers_iqr(df, outlier_columns_iqr)
    elif outlier_columns_medcouple:
        df = remove_outliers_medcouple(df, outlier_columns_medcouple)

    logger.info(f"✅ Готово. Размер финального датасета: {df.shape}")
    return df

