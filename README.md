# Home Credit Default Risk

Первая итерация проекта готовит доменные модели для стекинга через отдельную отложенную выборку.

Подробное описание реализованного алгоритма: [docs/domain_model_training.md](/Users/igor/Repositories/fintech_AI_adv/docs/domain_model_training.md)

## Что уже есть

- единый `development/holdout` split, который сохраняется в `artifacts/splits/`
- доменные feature builders для:
  - `application`
  - `bureau`
  - `previous_application`
  - `installments`
  - `pos_cash`
  - `credit_card`
- trainer доменных моделей на LightGBM
- CV внутри `development` для выбора гиперпараметров доменной модели
- сохранение моделей, метрик, важностей и предсказаний для `development`, `holdout`, `test`

## Быстрый старт

Установить зависимости:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Или просто запустить полный shell-скрипт, который сам создаст `.venv` и установит зависимости:

```bash
bash scripts/train_all_domains.sh
```

По умолчанию проект запускается в debug-режиме через `configs/debug.yaml`:

- `cv_folds = 1`, то есть CV отключён
- подбор гиперпараметров отключён
- уменьшено число boosting iterations для более быстрой отладки

Полный конфиг сохранён в [configs/base.yaml](/Users/igor/Repositories/fintech_AI_adv/configs/base.yaml).

Запустить обучение одного домена:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_domain.py --domain application
```

Явно указать полный конфиг:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_domain.py --config configs/base.yaml --domain application
```

Запустить обучение всех доменов:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_domain.py --domain all
```

Или одним shell-скриптом с логированием в консоль и файл:

```bash
bash scripts/train_all_domains.sh
```

Явно debug-режим:

```bash
bash scripts/train_all_domains.sh --debug
```

Полный запуск:

```bash
bash scripts/train_all_domains.sh --full
```

Логи сохраняются в `artifacts/logs/`.

## Логика обучения

- общий `train -> development + holdout`
- внутри `development` выполняется стратифицированное CV
- на CV подбирается лучшая конфигурация LightGBM по среднему `ROC-AUC`
- затем лучшая конфигурация переобучается на всём `development`
- финальная доменная оценка считается только на `holdout`

CV-результаты по каждому домену сохраняются в `artifacts/domain_models/<domain>/cv_results.csv`.
