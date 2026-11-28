# PythonPro2026

## О проекте

Пайплайн для хемоинформатического анализа набора молекул (на основе RDKit):

1. получает молекулы в формате SMILES из выбранного источника:
   * `csv` – CSV-файл с колонками `mol_id`, `smiles` (например, `data/molecules.csv`);
   * `reinvent` – генерирует молекулы моделью [REINVENT4](https://github.com/MolecularAI/REINVENT4);
2. проверяет валидность структур и считает дескрипторы: MW, LogP, TPSA, HBD, HBA, число вращаемых связей, число колец, QED;
3. оценивает соответствие правилу пяти Липински (`Ro5_pass`);
4. кластеризует молекулы алгоритмом Butina по сходству Танимото (Morgan-отпечатки, radius=2, 4096 бит);
5. строит TMAP-раскладку и сохраняет интерактивную визуализацию (Faerun) и таблицы результатов.

Структура:

```
main.py                  # точка входа: разбор аргументов и выбор источника
pipeline/
    graph_generator.py   # GraphGenerator: дескрипторы, кластеризация, граф, отчёты
    molecule.py          # Molecule: дескрипторы и фингерпринт одной молекулы
    sources/             # источники молекул: Source (ABC), CsvSource, ReinventSource
data/molecules.csv       # пример входных данных
output/                  # результаты (CSV и HTML-визуализация)
.env.example             # пример настроек для REINVENT
pyproject.toml, uv.lock  # зависимости
```

## Системные требования

- Python 3.12 или новее
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов и окружений
- Windows, Linux или macOS
- Основные зависимости (ставятся автоматически): `rdkit`, `tmap2`, `faerun`, `numpy`, `matplotlib<3.9`
- Для источника `reinvent` – установленный отдельно [REINVENT4](https://github.com/MolecularAI/REINVENT4) со своим окружением и файл модели (prior), например `priors/reinvent_pubchem.prior`
- Современный браузер для просмотра результата

## Инструкция по запуску

1. Клонируйте репозиторий и перейдите в папку проекта:

   ```bash
   git clone <URL репозитория>
   cd PythonPro2026
   ```

2. Создайте окружение и установите зависимости:

   ```bash
   uv sync
   ```

3. Запустите пайплайн на молекулах из CSV-файла:

   ```bash
   uv run python main.py csv data/molecules.csv
   ```

4. Чтобы генерировать молекулы REINVENT, скопируйте `.env.example` в `.env` и укажите в нём пути к `reinvent` и к файлу модели. Затем запустите:

   ```bash
   uv run --env-file .env python main.py reinvent
   ```

   Файл `.env` содержит пути вашей машины, поэтому он в `.gitignore` и не попадает в репозиторий.

5. Результаты появятся в папке `output/` (другую папку можно задать опцией `--output`, она пишется перед источником: `main.py --output results csv data/molecules.csv`):
   * `molecules_processed.csv` – валидные молекулы с дескрипторами;
   * `molecules_invalid.csv` – строки с невалидными SMILES (если такие есть);
   * `molecules.html` – визуализация, её нужно открыть в браузере.

Список всех опций: `uv run python main.py --help`.

## Контактные данные

- Автор: Kristina Sharaeva
- Email: kristinasharaeva1999@gmail.com
