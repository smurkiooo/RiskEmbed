# Запуск

Python 3.11-3.13, macOS/Linux. Для полного эксперимента ориентир: 8 ГБ RAM;
агрегаты считаются частями, последовательности хранятся компактно. GPU не требуется.
Сырые данные занимают около 1.7 ГБ, подготовленные данные и модели требуют дополнительного места.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
```

## Данные

1. Принять [правила AmEx](https://www.kaggle.com/competitions/amex-default-prediction).
2. Скачать официальный `train_labels.csv` и `train.parquet` из
   [raddar/amex-data-integer-dtypes-parquet-format](https://www.kaggle.com/datasets/raddar/amex-data-integer-dtypes-parquet-format).
3. Положить оба файла в `data/raw/amex/`. Не заменять parquet произвольной CSV-конвертацией:
   текущий адаптер ожидает целочисленные коды категорий и `-1` как пропуск в integer-колонках.

```bash
.venv/bin/python scripts/audit_amex.py
```

Аудит проверяет весь источник, в том числе перед быстрым экспериментом.
Проверки и SHA256 сохраняются в `reports/data_audit.json`.
Число строк подтверждает полноту выборки, но не побитовую эквивалентность исходному CSV.

## Эксперимент

```bash
.venv/bin/python scripts/run_experiment.py --quick --force
.venv/bin/python scripts/run_experiment.py --force
```

`--quick` выбирает примерно 3% клиентов и сокращает предобучение. Его результаты
хранятся отдельно и не подходят для выводов о качестве. `--force` пересчитывает энкодеры;
после изменения данных или параметров используйте его обязательно.

Отдельные этапы:

```bash
.venv/bin/python scripts/run_experiment.py --stage audit
.venv/bin/python scripts/run_experiment.py --stage prepare --quick
.venv/bin/python scripts/run_experiment.py --stage pretrain --quick --force
.venv/bin/python scripts/run_experiment.py --stage supervised --quick
.venv/bin/python scripts/run_experiment.py --stage boost --quick
.venv/bin/python scripts/run_experiment.py --stage analyze --quick
.venv/bin/python scripts/run_experiment.py --stage report --quick
```

Итог: `reports/EXPERIMENT.md` или `reports/EXPERIMENT_quick.md`.
Предобучение PyTorch и бустинг запускаются в разных процессах: это предотвращает
известный конфликт OpenMP на macOS.

## Airflow

Один DAG `amex_experiment`, запускаемый вручную:

```text
audit_data → prepare_data → pretrain_encoder → train_transformers → train_boosting → analyze_embeddings → build_report
```

DAG использует публичный `airflow.sdk` Airflow 3.1+. Airflow: необязательная зависимость
для локального CLI; для оркестратора установите его в отдельное окружение по
[официальной инструкции с constraints](https://airflow.apache.org/docs/apache-airflow/stable/start.html),
затем установите в это окружение данный проект (`pip install -e ".[dev]"`).
Версия Python и constraints должны совпадать. Предусмотрен optional extra `[airflow]`.

Пример переменных для локального запуска (из корня проекта):

```bash
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow standalone
```

В UI включите `amex_experiment` и запустите с `quick=true`, `force=true`.
Для полного запуска: `quick=false`. Все задачи должны видеть один каталог проекта,
его данные и установленное ML-окружение. Это локальный исследовательский DAG:
распределённое хранение и параллельные эксперименты здесь не реализованы.
`max_active_runs=1` исключает одновременные запуски этого DAG; не запускайте CLI одновременно с ним.

## Инференс

```bash
.venv/bin/python scripts/predict.py \
  --input data/raw/amex/train.parquet \
  --model artifacts/amex_full/transformer_finetuned \
  --output reports/predictions.csv
```

На входе integer-parquet той же схемы. Словарь берётся из пакета модели, не оценивается
по новым данным. На выходе одна строка `customer_ID,prediction` на клиента.
Это score для исходной прореженной выборки, не калиброванная банковская PD.

## Отчёты и проверка

`reports/EXPERIMENT.md`: таблицы; `reports/experiment.json`: данные отчёта;
`project_overview.html` в корне: автономная интерактивная визуализация с объяснениями.
Быстрый режим создаёт отдельный `project_overview_quick.html`. Открыть HTML можно прямо в браузере.
Сетевые запросы и внешние библиотеки для него не требуются.

Проверки: `pytest` и `ruff check src scripts dags tests`. Тест DAG пропускается без
Airflow; в окружении с Airflow 3.3.2 проверяется DAG.
Предобучение автоматически проверяет подписи входов и параметров перед чтением кеша.
Для полного пересчёта всё равно удобно использовать `--force`.

Полный запуск может занимать десятки минут на CPU; точное время каждого этапа
записывается в `reports/run_full.json`. Не запускайте два эксперимента одновременно.

## Проверка оркестратора

Помимо импорта DAG, быстрый эксперимент был выполнен через Airflow 3.3.2
методом `dag.test(run_conf={"quick": True, "force": True})` с локальной metadata DB.
Все семь задач завершились успешно. Интеграционная проверка охватывает выполнение DAG. Постоянный scheduler и UI
отдельно не тестировались.

При полном этапе `report` дополнительно выполняются независимый аудит сохранённых
предсказаний и shuffle-control на 3% реальных клиентов. sklearn для независимого пересчёта включён в основные зависимости.
Границы этих проверок: [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md).

## Проверенные запуски

- Полный CLI-эксперимент на 458 913 клиентах завершён, все семь этапов имеют статус passed.
- 30 тестов проходят в проверочном окружении с Airflow 3.3.2; Ruff без ошибок.
- Airflow DAG успешно выполнен в quick-режиме через `dag.test()`.
- HTML-страница проверена в Chromium на 320/390/768/1440 px, без ошибок JS
  и переполнения; интерактивные элементы проверены.
- Инференс из отдельного parquet с сохранённым пакетом модели проверен на 9 клиентах.
- Машиночитаемая сводка: `reports/validation.json`. Постоянный scheduler и UI не развёрнуты.
