#include "id3.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

// Подсчёт частот классов
static std::map<std::string, int> countLabels(const std::vector<Example>& data) {
    std::map<std::string, int> freq;
    for (const auto& ex : data) {
        ++freq[ex.label];
    }
    return freq;
}

static bool isPure(const std::vector<Example>& data) {
    if (data.empty()) return true;
    const std::string& firstLabel = data.front().label;
    for (const auto& ex : data) {
        if (ex.label != firstLabel) return false;
    }
    return true;
}

static std::string majorityClass(const std::vector<Example>& data) {
    auto freq = countLabels(data);
    if (freq.empty()) {
        return {};
    }
    return std::max_element(
               freq.begin(), freq.end(),
               [](const auto& a, const auto& b) {
                   return a.second < b.second;
               })
        ->first;
}

static double entropy(const std::vector<Example>& data) {
    auto freq = countLabels(data);
    if (freq.empty()) return 0.0;

    const double n = static_cast<double>(data.size());
    double h = 0.0;
    for (const auto& [label, count] : freq) {
        double p = count / n;
        if (p > 0.0) {
            h -= p * std::log2(p);
        }
    }
    return h;
}

// Разбиение выборки по значению одного атрибута
static std::map<std::string, std::vector<Example>>
splitByAttribute(const std::vector<Example>& data, int attrIndex) {
    std::map<std::string, std::vector<Example>> subsets;
    for (const auto& ex : data) {
        if (attrIndex < 0 || attrIndex >= static_cast<int>(ex.attrs.size())) continue;
        subsets[ex.attrs[attrIndex]].push_back(ex);
    }
    return subsets;
}

// Информационный выигрыш
static double informationGain(const std::vector<Example>& data, int attrIndex) {
    double baseEntropy = entropy(data);
    auto subsets = splitByAttribute(data, attrIndex);
    if (subsets.empty()) return 0.0;

    double n = static_cast<double>(data.size());
    double condEntropy = 0.0;

    for (const auto& [value, subset] : subsets) {
        double w = static_cast<double>(subset.size()) / n;
        condEntropy += w * entropy(subset);
    }

    return baseEntropy - condEntropy;
}

TreeNode* buildID3(const std::vector<Example>& data,
                   const std::vector<std::string>& attrNames,
                   const std::vector<int>& availableAttributes) {
    // Если выборка пустая — возвращаем пустой лист (на практике такого быть не должно)
    if (data.empty()) {
        auto* node = new TreeNode();
        node->isLeaf = true;
        node->label = "Нет данных";
        return node;
    }

    // Если все объекты одного класса — лист с этим классом
    if (isPure(data)) {
        auto* node = new TreeNode();
        node->isLeaf = true;
        node->label = data.front().label;
        return node;
    }

    // Если атрибутов не осталось — лист с наиболее частым классом
    if (availableAttributes.empty()) {
        auto* node = new TreeNode();
        node->isLeaf = true;
        node->label = majorityClass(data);
        return node;
    }

    // Выбираем атрибут с максимальным информационным выигрышем
    double bestGain = -1.0;
    int bestAttr = -1;

    for (int attrIndex : availableAttributes) {
        double gain = informationGain(data, attrIndex);
        if (gain > bestGain) {
            bestGain = gain;
            bestAttr = attrIndex;
        }
    }

    if (bestAttr == -1) {
        // На всякий случай — fallback: лист с majority class
        auto* node = new TreeNode();
        node->isLeaf = true;
        node->label = majorityClass(data);
        return node;
    }

    auto* node = new TreeNode();
    node->isLeaf = false;
    node->label = attrNames[bestAttr]; // имя признака

    auto subsets = splitByAttribute(data, bestAttr);

    // Новый список доступных атрибутов (без bestAttr)
    std::vector<int> newAvailable;
    newAvailable.reserve(availableAttributes.size() - 1);
    for (int idx : availableAttributes) {
        if (idx != bestAttr) newAvailable.push_back(idx);
    }

    // Строим поддеревья для каждого значения атрибута
    for (const auto& [value, subset] : subsets) {
        if (subset.empty()) {
            auto* child = new TreeNode();
            child->isLeaf = true;
            child->label = majorityClass(data);
            node->children[value] = child;
        } else {
            node->children[value] = buildID3(subset, attrNames, newAvailable);
        }
    }

    return node;
}

// Поиск индекса атрибута по имени
static int findAttributeIndex(const std::vector<std::string>& attrNames,
                              const std::string& name) {
    for (std::size_t i = 0; i < attrNames.size(); ++i) {
        if (attrNames[i] == name) return static_cast<int>(i);
    }
    return -1;
}

std::string classify(const TreeNode* root,
                     const Example& example,
                     const std::vector<std::string>& attrNames) {
    const TreeNode* node = root;

    while (node && !node->isLeaf) {
        int attrIndex = findAttributeIndex(attrNames, node->label);
        if (attrIndex == -1 ||
            attrIndex >= static_cast<int>(example.attrs.size())) {
            // не нашли атрибут — возвращаем majority class "по умолчанию"
            return "Неизвестно";
        }

        const std::string& val = example.attrs[attrIndex];
        auto it = node->children.find(val);
        if (it == node->children.end()) {
            // нет такого значения в дереве
            return "Неизвестно";
        }

        node = it->second;
    }

    if (!node) return "Неизвестно";
    return node->label;
}

void freeTree(TreeNode* root) {
    if (!root) return;
    for (auto& [value, child] : root->children) {
        freeTree(child);
    }
    delete root;
}
