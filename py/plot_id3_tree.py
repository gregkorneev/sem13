#!/usr/bin/env python3
import csv
import os
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


# ---------- ID3 на Python ----------

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


def majority_label(examples):
    freq = {}
    for ex in examples:
        freq[ex["label"]] = freq.get(ex["label"], 0) + 1
    return max(freq.items(), key=lambda x: x[1])[0]


class Node:
    def __init__(self, is_leaf=False, label=None, attr_index=None, attr_name=None):
        self.is_leaf = is_leaf
        self.label = label          # для листа: класс, для узла: имя атрибута
        self.attr_index = attr_index
        self.attr_name = attr_name
        self.children = {}          # value -> Node


def build_id3(examples, attr_names, available_attrs):
    # все одного класса?
    labels = {ex["label"] for ex in examples}
    if len(labels) == 1:
        return Node(is_leaf=True, label=next(iter(labels)))

    if not available_attrs or not examples:
        return Node(is_leaf=True, label=majority_label(examples))

    # выбираем атрибут с максимальным IG
    best_gain = -1.0
    best_attr = None
    for idx in available_attrs:
        gain = information_gain(examples, idx)
        if gain > best_gain:
            best_gain = gain
            best_attr = idx

    if best_attr is None or best_gain <= 1e-9:
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
            child = build_id3(subset, attr_names, new_available)
        node.children[value] = child

    return node


# ---------- раскладка дерева для matplotlib ----------

def count_leaves(node):
    if node.is_leaf or not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children.values())


def assign_positions(node, x_min, x_max, y, positions, parent=None, edges=None, value_labels=None):
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


def plot_tree(root, output_path="data/id3_tree.png"):
    positions = {}
    edges = []
    value_labels = {}

    assign_positions(root, x_min=0.0, x_max=1.0, y=0.0,
                     positions=positions, edges=edges, value_labels=value_labels)

    # подготавливаем папку
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # рисуем рёбра
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

    # рисуем узлы
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
    # по y делаем небольшой запас
    ys = [pos[1] for pos in positions.values()]
    ax.set_ylim(min(ys) - 0.5, 0.5)

    ax.axis("off")
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    print(f"Дерево сохранено в картинку: {output_path}")


# ---------- main ----------

def main():
    csv_path = os.path.join("data", "supplier_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Файл {csv_path} не найден. Сначала запусти C++ программу, "
            "чтобы она сгенерировала CSV."
        )

    attr_names, target_name, data = load_dataset(csv_path)
    print("Атрибуты:", attr_names)
    print("Целевой атрибут:", target_name)
    print(f"Количество примеров: {len(data)}")

    available_attrs = list(range(len(attr_names)))
    root = build_id3(data, attr_names, available_attrs)

    plot_tree(root, output_path=os.path.join("data", "id3_tree.png"))


if __name__ == "__main__":
    main()
