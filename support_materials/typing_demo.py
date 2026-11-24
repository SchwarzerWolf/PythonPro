"""Аннотации типов в Python: короткая демонстрация к лекции.

Запуск:            uv run python support_materials/typing_demo.py
Проверка типов:    uv run mypy support_materials/typing_demo.py
"""

from collections.abc import Callable
from typing import Any

from rdkit import Chem

# Python — язык с динамической типизацией
# Тип принадлежит значению, а не переменной: одну и ту же переменную
# можно перепривязать к значению другого типа.
value = 8
print(type(value))  # <class 'int'>
value = "8"  # type: ignore[assignment]  # mypy запомнил value как int
print(type(value))  # <class 'str'>


# Аннотации - подсказка, а не проверка
def double(number: int) -> int:
    """Удваивает число."""
    return number * 2


print(double(8))  # 16
# Интерпретатор аннотации не проверяет: строка "умножится" на 2 без ошибки.
# Ошибку найдёт только статический анализатор (mypy, PyCharm).
# Уберите комментарий type: ignore и запустите mypy, чтобы увидеть её.
print(double("88"))  # type: ignore[arg-type]  # -> '8888'


# Аннотации переменных
a: bool = True
b: int = 2
c: str = "something"
d: list[int] = [1, 2, 3]  # список целых чисел
e: dict[str, float] = {}  # для пустого словаря аннотация особенно полезна
f: list[dict[str, int]] = [{}, {}]  # вложенные типы
g: tuple[str, float] = ("MW", 180.16)  # кортеж фиксированной длины
h: tuple[int, ...] = (1, 2, 3)  # кортеж любой длины
i: set[str] = {"C", "N", "O"}


# Any - "любой тип", проверка отключается
# Линтер ruff (правило ANN401) ругается на Any - подавляем через noqa.
# Обратите внимание: noqa - для линтеров, type: ignore - для mypy.
def print_everything(item: Any) -> None:  # noqa: ANN401
    """Печатает что угодно."""
    print(item)


def shout(item: Any) -> str:  # noqa: ANN401
    """С Any mypy молчит, даже если у объекта нет такого метода."""
    return item.upper()  # упадёт при вызове с int, но mypy не предупредит


def shout_safe(item: object) -> str:
    """Принимает любой объект, но mypy требует проверки типа."""
    # return item.upper()  # mypy: "object" has no attribute "upper"
    if isinstance(item, str):
        return item.upper()  # здесь mypy уже знает, что item — str
    return str(item)


print_everything([1, "two", 3.0])
print(shout_safe("benzene"), shout_safe(42))


# Optional: значение может отсутствовать
# Chem.MolFromSmiles возвращает None для некорректного SMILES.
# Optional[X] из typing и X | None (Python 3.10+) — одно и то же.
def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Возвращает молекулу или None, если SMILES некорректен."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        print(f"Некорректный SMILES: {smiles}")
        return None  # mypy потребует явный return во всех ветках
    return molecule


def count_atoms(smiles: str) -> int:
    """Считает тяжёлые атомы; показывает сужение типа (type narrowing)."""
    molecule = parse_smiles(smiles)
    # molecule.GetNumAtoms()  # mypy: Item "None" has no attribute "GetNumAtoms"
    if molecule is None:
        return 0
    return molecule.GetNumAtoms()  # после проверки тип — Chem.Mol


print(count_atoms("c1ccccc1"))  # 6
print(count_atoms("not_a_smiles"))  # 0


def to_float(value: int | str) -> float:
    """Принимает int или str."""
    return float(value)


def apply_to_all(func: Callable[[str], int], items: list[str]) -> list[int]:
    """Применяет функцию к каждому элементу списка."""
    return [func(item) for item in items]


print(apply_to_all(count_atoms, ["CCO", "c1ccccc1"]))  # [3, 6]


# Псевдонимы типов и обобщённые функции (Python 3.12+)
type Descriptors = dict[str, float]  # имя для длинного типа


def mol_weight(descriptors: Descriptors) -> float:
    """Достаёт молекулярную массу из словаря дескрипторов."""
    return descriptors["MW"]


def first[T](items: list[T]) -> T:
    """Тип результата совпадает с типом элементов списка."""
    return items[0]


print(mol_weight({"MW": 46.07, "LogP": -0.14}))
print(first(["CCO", "CCN"]))  # mypy знает, что это str
print(first([1, 2, 3]))  # а здесь — int
