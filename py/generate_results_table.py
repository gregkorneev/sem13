#!/usr/bin/env python3
"""
Генерация сводной таблицы результатов для алгоритмов
ID3, C4.5, CART, CHAID по данным из supplier_dataset.csv.

Делаем train/test split (по умолчанию 10/4) и считаем
Точность и F1 только на тестовой выборке.
"""

import time
import math
import random
from collections import Counter, defaultdict

import pandas as pd


# ---------- Чтение данных ----------

def load_dataset(path: str = "data/supplier_dataset.csv"):
    df = pd.read_csv(path, sep=";")
    columns = list(df.columns)
    target_col = columns[-1]
    attr_names = columns[:-1]

    examples = []
    for _, row in df.iterrows():
        attrs = {col: row[col] for col in attr_names}
        label = row[target_col]
        examples.append((attrs, label))

    return attr_names, target_col, examples


# ---------- Train / test split ----------

def train_test_split(examples, test_size=4, seed=42):
    """
    Простое разбиение 10/4 (по умолчанию) с фиксированным seed
    для воспроизводимости.
    """
    rnd = random.Random(seed)
    indices = list(range(len(examples)))
    rnd.shuffle(indices)

    if isinstance(test_size, float):
        k_test = max(1, int(round(len(examples) * test_size)))
    else:
        k_test = min(len(examples) - 1, int(test_size))

    test_idx = set(indices[:k_test])
    train, test = [], []
    for i, ex in enumerate(examples):
        if i in test_idx:
            test.append(ex)
        else:
            train.append(ex)

    return train, test


# ---------- Структура узла дерева ----------

class TreeNode:
    def __init__(self, is_leaf=False, label=None, attr=None):
        self.is_leaf = is_leaf
        self.label = label
        self.attr = attr
        self.children = {}

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.label})"
        return f"Node({self.attr}, children={list(self.children.keys())})"


# ---------- Вспомогательные функции для критериев ----------

def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counter = Counter(labels)
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def gini(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counter = Counter(labels)
    g = 1.0
    for c in counter.values():
        p = c / total
        g -= p * p
    return g


def split_by_attr(examples, attr):
    buckets = defaultdict(list)
    for attrs, label in examples:
        buckets[attrs[attr]].append((attrs, label))
    return buckets


def information_gain(examples, attr):
    labels = [label for _, label in examples]
    base_h = entropy(labels)
    buckets = split_by_attr(examples, attr)
    total = len(examples)
    cond_h = 0.0
    for subset in buckets.values():
        w = len(subset) / total
        cond_h += w * entropy([l for _, l in subset])
    return base_h - cond_h


def split_info(examples, attr):
    buckets = split_by_attr(examples, attr)
    total = len(examples)
    si = 0.0
    for subset in buckets.values():
        p = len(subset) / total
        if p > 0:
            si -= p * math.log2(p)
    return si


def gain_ratio(examples, attr):
    ig = information_gain(examples, attr)
    si = split_info(examples, attr)
    if si == 0:
        return 0.0
    return ig / si


def gini_gain(examples, attr):
    labels = [label for _, label in examples]
    base_g = gini(labels)
    buckets = split_by_attr(examples, attr)
    total = len(examples)
    cond_g = 0.0
    for subset in buckets.values():
        w = len(subset) / total
        cond_g += w * gini([l for _, l in subset])
    return base_g - cond_g


def chi_square_score(examples, attr):
    buckets = split_by_attr(examples, attr)
    if not buckets:
        return 0.0

    labels = [label for _, label in examples]
    total = len(examples)
    label_counts = Counter(labels)
    chi2 = 0.0

    for subset in buckets.values():
        subset_size = len(subset)
        subset_labels = [l for _, l in subset]
        subset_counts = Counter(subset_labels)
        for cls, total_cls_count in label_counts.items():
            expected = total_cls_count * subset_size / total
            observed = subset_counts.get(cls, 0)
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
    return chi2


# ---------- Построение дерева ----------

def majority_label(examples):
    labels = [label for _, label in examples]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def build_tree(examples, attr_names, algo: str):
    labels = [label for _, label in examples]
    if len(set(labels)) == 1:
        return TreeNode(is_leaf=True, label=labels[0])

    if not attr_names:
        return TreeNode(is_leaf=True, label=majority_label(examples))

    best_attr = None
    best_score = -1.0

    for attr in attr_names:
        if algo == "id3":
            score = information_gain(examples, attr)
        elif algo == "c45":
            score = gain_ratio(examples, attr)
        elif algo == "cart":
            score = gini_gain(examples, attr)
        elif algo == "chaid":
            score = chi_square_score(examples, attr)
        else:
            raise ValueError(f"Неизвестный алгоритм: {algo}")

        if score > best_score:
            best_score = score
            best_attr = attr

    if best_attr is None:
        return TreeNode(is_leaf=True, label=majority_label(examples))

    node = TreeNode(is_leaf=False, attr=best_attr)
    buckets = split_by_attr(examples, best_attr)
    remaining_attrs = [a for a in attr_names if a != best_attr]

    for value, subset in buckets.items():
        if not subset:
            child = TreeNode(is_leaf=True, label=majority_label(examples))
        else:
            child = build_tree(subset, remaining_attrs, algo)
        node.children[value] = child

    return node


# ---------- Оценка дерева ----------

def predict_one(node: TreeNode, attrs: dict, default_label=None):
    while not node.is_leaf:
        attr = node.attr
        val = attrs.get(attr)
        if val in node.children:
            node = node.children[val]
        else:
            return default_label
    return node.label


def evaluate_algorithm(train_examples, test_examples, attr_names,
                       algo: str, positive_label="Да"):
    start = time.perf_counter()
    tree = build_tree(train_examples, attr_names, algo)
    elapsed = time.perf_counter() - start

    if not test_examples:
        return 0.0, 0.0, elapsed, 0

    default_label = majority_label(train_examples)
    y_true, y_pred = [], []

    for attrs, label in test_examples:
        pred = predict_one(tree, attrs, default_label=default_label)
        if pred is None:
            pred = default_label
        y_true.append(label)
        y_pred.append(pred)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)

    if tp == 0 and (fp > 0 or fn > 0):
        f1 = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    def count_nodes(n: TreeNode):
        if n is None:
            return 0
        total = 1
        for ch in n.children.values():
            total += count_nodes(ch)
        return total

    nodes = count_nodes(tree)
    return accuracy, f1, elapsed, nodes


# ---------- Главная функция ----------

def main():
    attr_names, target_col, examples = load_dataset()
    train_examples, test_examples = train_test_split(examples, test_size=4, seed=42)

    print(f"Всего объектов: {len(examples)} | train: {len(train_examples)}, test: {len(test_examples)}\n")

    algo_map = [
        ("ID3", "id3"),
        ("C4.5", "c45"),
        ("CART", "cart"),
        ("CHAID", "chaid"),
    ]

    rows = []
    for display_name, algo_code in algo_map:
        acc, f1, elapsed, nodes = evaluate_algorithm(
            train_examples, test_examples, attr_names, algo_code
        )
        rows.append({
            "Алгоритм": display_name,
            "Точность": round(acc, 4),
            "F1-Score": round(f1, 4),
            "Время (с)": round(elapsed, 6),
            "Узлы": int(nodes),
        })

    df = pd.DataFrame(rows, columns=["Алгоритм", "Точность", "F1-Score", "Время (с)", "Узлы"])

    print("Результаты (оценка по тестовой выборке):")
    print(df.to_string(index=False))

    df.to_csv("data/algorithms_summary.csv", index=False, encoding="utf-8-sig")
    print("\nТаблица сохранена в data/algorithms_summary.csv")


if __name__ == "__main__":
    main()
