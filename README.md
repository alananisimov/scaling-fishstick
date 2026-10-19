# Инструкция участника Green Bank Pipeline CTF

## Что нужно сделать

Вы создаёте репозиторий с CLI `python -m bank_ctf`, который обучает и применяет ML-пайплайн для двух задач:

1. `inflow`: предсказать `inflow_next_30d` для клиента;
2. `risk`: предсказать `overdue_next_60d` для кредитной заявки, используя кредитные признаки и честный прогноз `predicted_inflow`.

Проверяется не стиль кода, а поведение команд и артефактов в чистом окружении.

## Данные

Папка данных содержит:

| Файл | Назначение |
| --- | --- |
| `clients_train.csv` | клиентские признаки и target `inflow_next_30d` |
| `clients_test.csv` | клиентские признаки без target |
| `credit_train.csv` | кредитные признаки и target `overdue_next_60d` |
| `credit_test.csv` | кредитные признаки без target |
| `folds_random.csv` | начальные фолды по `client_id` |
| `folds_group.csv` | фолды по отделениям `branch_id` для этапа 4 |

`client_id` и `application_id` — только ключи соединения, не признаки модели.

## Обязательные команды

```bash
python -m bank_ctf train --task inflow --config configs/inflow.json --data-dir /data --output-dir /artifacts/inflow
python -m bank_ctf predict --task inflow --model-dir /artifacts/inflow --data-dir /data --output /artifacts/inflow_predictions.csv
python -m bank_ctf train --task risk --config configs/risk.json --data-dir /data --output-dir /artifacts/risk
python -m bank_ctf predict --task risk --model-dir /artifacts/risk --data-dir /data --output /artifacts/risk_predictions.csv
python -m bank_ctf experiment --config configs/experiment.json --data-dir /data --runs-dir /artifacts/runs --run-id baseline_group
```

Все пути должны работать из любого текущего каталога. Не ищите данные или модели в соседних папках неявно.

## Форматы выходов

### `inflow_predictions.csv`

```csv
client_id,predicted_inflow
T000000,2631.42
T000001,1510.03
```

Требования:

- один ряд на каждый `client_id` из `clients_test.csv`;
- порядок строк совпадает с `clients_test.csv`;
- `predicted_inflow` конечный и `>= 0`.

### `risk_predictions.csv`

```csv
application_id,probability_overdue
Z000000,0.1842
Z000001,0.0731
```

Требования:

- один ряд на каждый `application_id` из `credit_test.csv`;
- порядок строк совпадает с `credit_test.csv`;
- `probability_overdue` конечная и в `[0, 1]`.

### Экспериментальные артефакты

`experiment` должен записывать:

- `oof_inflow.csv`: `client_id,fold_id,predicted_inflow`;
- `oof_risk.csv`: `application_id,fold_id,probability_overdue`;
- `cv.json`: задача, метрика, split, fold scores, mean, std, seed;
- `manifest.json`: effective config, hashes, versions, model files;
- `runs.jsonl`: одна итоговая запись на run, включая `failed` runs;
- финальную модель и test-предсказания.

## Этапы

### Этап 1 — inflow baseline

Сделайте `train/predict --task inflow`. Проверки меняют размер test, порядок строк и порядок колонок. Модель должна реально обучаться и быть лучше простого константного baseline на hidden data.

### Этап 2 — risk chain

`train --task risk` должен сам обучить inflow-модель, получить OOF `predicted_inflow` для train-клиентов, обучить риск-модель и сохранить полный bundle. На test сначала применяется сохранённая inflow-модель, затем risk-модель.

Запрещено:

- использовать `inflow_next_30d` напрямую как признак риска;
- обучить inflow на всём train и предсказать тот же train для риска;
- соединять таблицы по позиции строк.

### Этап 3 — experiments

Поддержите JSON-конфиги, effective config, журнал `runs.jsonl`, повторяемые фолды и ошибку при повторном `run_id`. Неверный config должен завершаться ненулевым exit code с понятным сообщением.

### Этап 4 — новые отделения

Используйте `folds_group.csv` и `split_version=group_v1`. Все строки одного `branch_id` должны быть в одном outer fold. Для risk CV нужно изолировать всю цепочку: validation-fold не участвует ни в inflow fit, ни в preprocessing, ни в inner OOF.

### Этап 5 — чистый релиз

Checker клонирует фиксированный коммит и запускает всё без старых артефактов и без сети. Сохранённые модели должны переноситься в другой каталог и работать после удаления воспроизводимого кеша.

## Как проверить себя локально

```bash
uv sync --group dev
uv run python -m bank_ctf.generate_data --output-dir data/sample
uv run pytest -q
uv run python -m checker.open_checker --json-report artifacts/open_checker_report.json
```

Шаблон содержит рабочий stage-1 baseline: `train/predict --task inflow` можно запустить сразу. Остальные команды (`risk` и `experiment`) являются следующими этапами и должны быть реализованы участником. Генератор данных нужен только для локальных smoke/open checks; checker на соревновании передаст свою папку `/data`.

## Отправка решения

1. Нажмите **Use this template** или сделайте fork этого репозитория.
2. Реализуйте следующие этапы в своём fork и отправьте commit в `main`.
3. Зарегистрируйте URL репозитория на платформе соревнования. Платформа обнаружит новые commit и запустит checker в чистом рабочем каталоге.
4. Результаты и причины ошибок доступны в UI платформы; повторная проверка одного и того же commit/version tuple переиспользует сохранённый результат.

Не добавляйте в репозиторий private labels, секреты checker или токены GitHub. Платформа передаёт тестовые данные во время проверки и не ожидает их commit в fork.

Публичный шаблон не содержит `AGENTS.md`, `SKILLS.md`, `solutions/`, organizer notes, private labels, runner history или секреты выдачи флагов.

Если open checker проходит, это не гарантирует private score, но подтверждает базовый контракт.
