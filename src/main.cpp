#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <filesystem>

// ----------------------------
// Структуры данных
// ----------------------------

struct Example {
    std::vector<std::string> attrs; // значения атрибутов
    std::string label;              // целевой класс (Да/Нет)
};

struct Node {
    bool isLeaf = false;
    int attrIndex = -1;                   // индекс атрибута для разбиения, -1 для листа
    std::string label;                    // для листа: класс; для узла — имя атрибута
    std::map<std::string, std::unique_ptr<Node>> children; // значение атрибута -> поддерево
};

// ----------------------------
// Вспомогательные функции
// ----------------------------

// наиболее частый класс
std::string majorityLabel(const std::vector<Example>& data) {
    std::map<std::string, int> freq;
    for (const auto& ex : data) {
        ++freq[ex.label];
    }
    int bestCount = -1;
    std::string bestLabel;
    for (const auto& [lab, cnt] : freq) {
        if (cnt > bestCount) {
            bestCount = cnt;
            bestLabel = lab;
        }
    }
    return bestLabel;
}

// энтропия целевого атрибута
double entropy(const std::vector<Example>& data) {
    if (data.empty()) return 0.0;
    std::map<std::string, int> freq;
    for (const auto& ex : data) {
        ++freq[ex.label];
    }
    double result = 0.0;
    const double n = static_cast<double>(data.size());
    for (const auto& [lab, cnt] : freq) {
        double p = cnt / n;
        if (p > 0) {
            result -= p * std::log2(p);
        }
    }
    return result;
}

// разбиение по атрибуту
std::map<std::string, std::vector<Example>> splitByAttribute(
        const std::vector<Example>& data,
        int attrIndex) {
    std::map<std::string, std::vector<Example>> subsets;
    for (const auto& ex : data) {
        if (attrIndex < 0 || attrIndex >= (int)ex.attrs.size()) continue;
        subsets[ex.attrs[attrIndex]].push_back(ex);
    }
    return subsets;
}

// прирост информации (ID3)
double informationGain(const std::vector<Example>& data, int attrIndex) {
    if (data.empty()) return 0.0;
    double baseEntropy = entropy(data);
    auto subsets = splitByAttribute(data, attrIndex);

    double condEntropy = 0.0;
    double n = static_cast<double>(data.size());
    for (const auto& [value, subset] : subsets) {
        double weight = subset.size() / n;
        condEntropy += weight * entropy(subset);
    }
    return baseEntropy - condEntropy;
}

// все ли примеры одного класса
bool allSameLabel(const std::vector<Example>& data, std::string& labelOut) {
    if (data.empty()) return false;
    labelOut = data.front().label;
    for (const auto& ex : data) {
        if (ex.label != labelOut) return false;
    }
    return true;
}

// ----------------------------
// Сохранение таблицы в CSV
// ----------------------------

void saveDatasetToCSV(const std::string& filename,
                      const std::vector<Example>& data,
                      const std::vector<std::string>& attrNames) {
    namespace fs = std::filesystem;

    // создаём папку, если её нет
    fs::path path(filename);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Не удалось открыть файл " << filename << " для записи\n";
        return;
    }

    // заголовок
    for (size_t i = 0; i < attrNames.size(); ++i) {
        out << attrNames[i] << ";";
    }
    out << "Решение\n";

    // строки данных
    for (const auto& ex : data) {
        for (size_t i = 0; i < ex.attrs.size(); ++i) {
            out << ex.attrs[i] << ";";
        }
        out << ex.label << "\n";
    }

    std::cout << "Таблица обучающей выборки сохранена в CSV: " << filename << "\n";
}

// ----------------------------
// Построение дерева ID3
// ----------------------------

std::unique_ptr<Node> buildID3(
    const std::vector<Example>& data,
    const std::vector<std::string>& attrNames,
    const std::vector<int>& availableAttrs
) {
    auto node = std::make_unique<Node>();

    std::string singleLabel;
    if (allSameLabel(data, singleLabel)) {
        node->isLeaf = true;
        node->label = singleLabel;
        return node;
    }

    if (availableAttrs.empty() || data.empty()) {
        node->isLeaf = true;
        node->label = majorityLabel(data);
        return node;
    }

    double bestGain = -1.0;
    int bestAttrIndex = -1;

    for (int idx : availableAttrs) {
        double gain = informationGain(data, idx);
        if (gain > bestGain) {
            bestGain = gain;
            bestAttrIndex = idx;
        }
    }

    if (bestAttrIndex == -1 || bestGain <= 1e-9) {
        node->isLeaf = true;
        node->label = majorityLabel(data);
        return node;
    }

    node->isLeaf = false;
    node->attrIndex = bestAttrIndex;
    node->label = attrNames[bestAttrIndex];

    std::vector<int> newAvailable = availableAttrs;
    newAvailable.erase(
        std::remove(newAvailable.begin(), newAvailable.end(), bestAttrIndex),
        newAvailable.end()
    );

    auto subsets = splitByAttribute(data, bestAttrIndex);
    for (const auto& [value, subset] : subsets) {
        if (subset.empty()) {
            auto leaf = std::make_unique<Node>();
            leaf->isLeaf = true;
            leaf->label = majorityLabel(data);
            node->children[value] = std::move(leaf);
        } else {
            node->children[value] = buildID3(subset, attrNames, newAvailable);
        }
    }

    return node;
}

// ----------------------------
// Печать дерева
// ----------------------------

void printTree(const Node* node,
               const std::vector<std::string>& /*attrNames*/,
               const std::string& indent = "",
               const std::string& branchValue = "") {
    if (!node) return;

    std::string prefix = indent;
    if (!branchValue.empty()) {
        prefix += branchValue + " -> ";
    }

    if (node->isLeaf) {
        std::cout << prefix << "[КЛАСС: " << node->label << "]\n";
    } else {
        std::cout << prefix << "[АТРИБУТ: " << node->label << "]\n";
        std::string newIndent = indent + "    ";
        for (const auto& [value, child] : node->children) {
            printTree(child.get(), {},
                      newIndent, node->label + " = " + value);
        }
    }
}

// классификация примера
std::string classify(const Node* node,
                     const std::vector<std::string>& attrs) {
    const Node* cur = node;
    while (cur && !cur->isLeaf) {
        int idx = cur->attrIndex;
        auto it = cur->children.find(attrs[idx]);
        if (it == cur->children.end()) {
            return "Нет";
        }
        cur = it->second.get();
    }
    if (!cur) return "Нет";
    return cur->label;
}

// ----------------------------
// main
// ----------------------------

int main() {
    setlocale(LC_ALL, "");

    std::vector<std::string> attrNames = {
        "Цена",
        "Качество",
        "Срок поставки",
        "Надёжность"
    };

    std::vector<Example> data = {
        {{"низкая",  "высокое", "быстрая",   "высокая"}, "Да"},
        {{"средняя", "высокое", "нормальная","высокая"}, "Да"},
        {{"высокая", "высокое", "быстрая",   "высокая"}, "Нет"},
        {{"низкая",  "среднее","медленная", "средняя"}, "Да"},
        {{"низкая",  "низкое", "медленная", "низкая"},  "Нет"},
        {{"средняя", "среднее","нормальная","средняя"}, "Да"},
        {{"высокая", "среднее","медленная", "средняя"}, "Нет"},
        {{"средняя", "низкое", "быстрая",   "высокая"}, "Нет"},
        {{"низкая",  "высокое","медленная", "средняя"}, "Да"},
        {{"средняя", "высокое","медленная", "низкая"},  "Нет"},
        {{"высокая", "высокое","нормальная","средняя"}, "Нет"},
        {{"низкая",  "среднее","нормальная","высокая"}, "Да"},
        {{"средняя", "среднее","быстрая",   "низкая"},  "Нет"},
        {{"низкая",  "низкое", "нормальная","средняя"}, "Нет"}
    };

    std::vector<int> availableAttrs = {0, 1, 2, 3};

    // сохраняем обучающую выборку в CSV
    saveDatasetToCSV("data/supplier_dataset.csv", data, attrNames);

    // строим дерево ID3
    std::unique_ptr<Node> root = buildID3(data, attrNames, availableAttrs);

    std::cout << "\nДерево решений (алгоритм ID3) для задачи выбора поставщика:\n\n";
    printTree(root.get(), attrNames);

    std::vector<std::string> newSupplier = {"средняя", "высокое", "быстрая", "высокая"};
    std::string predicted = classify(root.get(), newSupplier);
    std::cout << "\nКлассификация нового поставщика "
              << "(Цена=средняя, Качество=высокое, Срок=быстрая, Надёжность=высокая): "
              << predicted << "\n";

    return 0;
}
