#!/usr/bin/env python3
import csv
import os
import sys
from math import log2
import matplotlib.pyplot as plt

# ---------- чтение данных из CSV ----------

def load_dataset(csv_path):
    """
    Ожидается формат:
    Цена;Качество;Срок поставки;Надёжность;Решение
    низкая;высокое;быстрая;высокая;Да
    ...
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=';')
        rows = list(reader)

    header = rows[0]
    data_rows = rows[1:]

    attr_names = header[:-1]   # все, кроме последнего
    target_name = header[-1]   # "Решение"

    data = []
    for row in data_rows:
        if not row:
            continue
        attrs = row[:-1]
        label = row[-1]
        data.append({"attrs": attrs, "label": label})

    return attr_names, target_name, data


# ---------- общие функции ----------

def entropy(examples):
    if not examples:
        return 0.0
    freq = {}
    for ex in examples:
        freq[ex["label"]] = freq.get(ex["label"], 0) + 1
    n = len(examples)
    result = 0.0
    for cnt in freq.values():
        p = cnt / n
        if p > 0:
            result -= p * log2(p)
    return result


def split_by_attr(examples, attr_index):
    subsets = {}
    for ex in examples:
        value = ex["attrs"][attr_index]
        subsets.setdefault(value, []).append(ex)
    return subsets


def information_gain(examples, attr_index):
    if not examples:
        return 0.0
    base = entropy(examples)
    subsets = split_by_attr(examples, attr_index)
    n = len(examples)
    cond = 0.0
    for subset in subsets.values():
        w = len(subset) / n
        cond += w * entropy(subset)
    return base - cond


def split_info(examples, attr_index):
    """
    SplitInfo для C4.5: энтропия распределения по значениям атрибута.
    """
    subsets = split_by_attr(examples, attr_index)
    n = len(examples)
    result = 0.0
    for subset in subsets.values():
        p = len(subset) / n
        if p > 0:
            result -= p * log2(p)
    return result


def gain_ratio(examples, attr_index):
    ig = information_gain(examples, attr_index)
    si = split_info(examples, attr_index)
    if si <= 1e-12:
        return 0.0
    return ig / si


def gini_impurity(examples):
    if not examples:
        return 0.0
    freq = {}
    for ex in examples:
        freq[ex["label"]] = freq.get(ex["label"], 0) + 1
    n = len(examples)
    result = 1.0
    for cnt in freq.values():
        p = cnt / n
        result -= p * p
    return result


def gini_gain(examples, attr_index):
    """
    Уменьшение Gini при разбиении по атрибуту (вариант CART).
    """
    if not examples:
        return 0.0
    base = gini_impurity(examples)
    subsets = split_by_attr(examples, attr_index)
    n = len(examples)
    cond = 0.0
    for subset in subsets.values():
        w = len(subset) / n
        cond += w * gini_impurity(subset)
    return base - cond


def chi_square_score(examples, attr_index):
    """
    Упрощённый CHAID: используем статистику χ² без p-value.
    Строим таблицу: значение атрибута x класс.
    """
    subsets = split_by_attr(examples, attr_index)
    if not subsets:
        return 0.0

    # множество всех классов
    classes = sorted({ex["label"] for ex in examples})
    class_index = {c: i for i, c in enumerate(classes)}

    # строим таблицу наблюдаемых частот
    values = list(subsets.keys())
    value_index = {v: i for i, v in enumerate(values)}

    rows = len(values)
    cols = len(classes)

    # O[i][j] — наблюдаемое количество
    O = [[0 for _ in range(cols)] for _ in range(rows)]
    for v, subset in subsets.items():
        i = value_index[v]
        for ex in subset:
            j = class_index[ex["label"]]
            O[i][j] += 1

    # суммы по строкам/столбцам
    row_sum = [sum(O[i][j] for j in range(cols)) for i in range(rows)]
    col_sum = [sum(O[i][j] for i in range(rows)) for j in range(cols)]
    total = sum(row_sum)
    if total == 0:
        return 0.0

    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            expected = row_sum[i] * col_sum[j] / total
            if expected > 0:
                chi2 += (O[i][j] - expected) ** 2 / expected

    return chi2


def majority_label(examples):
    freq = {}
    for ex in examples:
        freq[ex["label"]] = freq.get(ex["label"], 0) + 1
    return max(freq.items(), key=lambda x: x[1])[0]


class Node:
    def __init__(self, is_leaf=False, label=None, attr_index=None, attr_name=None):
        self.is_leaf = is_leaf
        self.label = label          # для листа: класс; для узла: имя атрибута
        self.attr_index = attr_index
        self.attr_name = attr_name
        self.children = {}          # value -> Node


# ---------- построение дерева для разных алгоритмов ----------

def choose_best_attribute(examples, attr_names, available_attrs, algo: str):
    """
    Возвращает индекс лучшего атрибута по выбранному алгоритму.
    algo in {"id3", "c45", "cart", "chaid"}
    """
    best_score = -1.0
    best_attr = None

    for idx in available_attrs:
        if algo == "id3":
            score = information_gain(examples, idx)
        elif algo == "c45":
            score = gain_ratio(examples, idx)
        elif algo == "cart":
            score = gini_gain(examples, idx)
        elif algo == "chaid":
            score = chi_square_score(examples, idx)
        else:
            raise ValueError(f"Неизвестный алгоритм: {algo}")

        if score > best_score:
            best_score = score
            best_attr = idx

    # порог на "слишком маленький" выигрыш
    if best_attr is None or best_score <= 1e-9:
        return None
    return best_attr


def build_tree(examples, attr_names, available_attrs, algo: str):
    """
    Единый конструктор дерева для ID3 / C4.5 / CART / CHAID.
    """
    # все примеры одного класса -> лист
    labels = {ex["label"] for ex in examples}
    if len(labels) == 1:
        return Node(is_leaf=True, label=next(iter(labels)))

    # нет атрибутов или пустое множество -> лист с majority class
    if not available_attrs or not examples:
        return Node(is_leaf=True, label=majority_label(examples))

    best_attr = choose_best_attribute(examples, attr_names, available_attrs, algo)
    if best_attr is None:
        return Node(is_leaf=True, label=majority_label(examples))

    node = Node(
        is_leaf=False,
        label=attr_names[best_attr],
        attr_index=best_attr,
        attr_name=attr_names[best_attr]
    )

    new_available = [i for i in available_attrs if i != best_attr]
    subsets = split_by_attr(examples, best_attr)

    for value, subset in subsets.items():
        if not subset:
            child = Node(is_leaf=True, label=majority_label(examples))
        else:
            child = build_tree(subset, attr_names, new_available, algo)
        node.children[value] = child

    return node


# ---------- раскладка дерева для matplotlib ----------

def count_leaves(node):
    if node.is_leaf or not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children.values())


def assign_positions(node, x_min, x_max, y, positions,
                     parent=None, edges=None, value_labels=None):
    """
    Рекурсивно назначаем координаты узлам.
    x_min, x_max – горизонтальный диапазон для поддерева.
    y – уровень по вертикали (0, -1, -2, ...).
    """
    if edges is None:
        edges = []
    if value_labels is None:
        value_labels = {}

    x = (x_min + x_max) / 2.0
    positions[node] = (x, y)

    if not node.children:
        return

    total_leaves = sum(count_leaves(child) for child in node.children.values())
    cur_x = x_min
    for value, child in node.children.items():
        leaves = count_leaves(child)
        width = (x_max - x_min) * (leaves / total_leaves)
        child_x_min = cur_x
        child_x_max = cur_x + width
        cur_x += width

        # ребро родитель -> ребёнок
        edges.append((node, child))
        value_labels[(node, child)] = value

        assign_positions(child, child_x_min, child_x_max, y - 1,
                         positions, node, edges, value_labels)


def plot_tree(root, output_path="data/tree.png", title=None):
    positions = {}
    edges = []
    value_labels = {}

    assign_positions(root, x_min=0.0, x_max=1.0, y=0.0,
                     positions=positions, edges=edges, value_labels=value_labels)

    # подготавливаем папку
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # рёбра
    for parent, child in edges:
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        ax.plot([x1, x2], [y1, y2])

        # подпись ветви (значение атрибута)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        text = value_labels[(parent, child)]
        ax.text(mid_x, mid_y + 0.03, text,
                ha="center", va="bottom")

    # узлы
    for node, (x, y) in positions.items():
        if node.is_leaf:
            circle = plt.Circle((x, y), 0.02, fill=True, alpha=0.3)
            ax.add_patch(circle)
            ax.text(x, y, node.label, ha="center", va="center")
        else:
            rect = plt.Rectangle((x - 0.04, y - 0.02), 0.08, 0.04,
                                 fill=True, alpha=0.3)
            ax.add_patch(rect)
            ax.text(x, y, node.attr_name, ha="center", va="center")

    ax.set_xlim(-0.05, 1.05)
    ys = [pos[1] for pos in positions.values()]
    ax.set_ylim(min(ys) - 0.5, 0.5)

    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    print(f"Дерево сохранено в картинку: {output_path}")


# ---------- main ----------

def main():
    if len(sys.argv) >= 2:
        algo = sys.argv[1].lower()
    else:
        algo = "id3"

    if algo not in {"id3", "c45", "cart", "chaid"}:
        raise SystemExit("Использование: python3 plot_id3_tree.py [id3|c45|cart|chaid]")

    csv_path = os.path.join("data", "supplier_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Файл {csv_path} не найден. Сначала запусти C++ программу,"
            " чтобы она сгенерировала CSV."
        )

    attr_names, target_name, data = load_dataset(csv_path)
    print("Атрибуты:", attr_names)
    print("Целевой атрибут:", target_name)
    print(f"Количество примеров: {len(data)}")
    print(f"Алгоритм: {algo.upper()}")

    available_attrs = list(range(len(attr_names)))
    root = build_tree(data, attr_names, available_attrs, algo)

    output_name = f"{algo}_tree.png"
    output_path = os.path.join("data", output_name)
    title = f"Дерево решений ({algo.upper()})"
    plot_tree(root, output_path=output_path, title=title)


if __name__ == "__main__":
    main()
