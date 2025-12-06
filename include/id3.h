#pragma once

#include "dataset.h"

#include <map>
#include <string>
#include <vector>

// Узел дерева решений
struct TreeNode {
    bool isLeaf = false;                       // true, если лист
    std::string label;                         // если лист — класс; если нет — имя атрибута
    std::map<std::string, TreeNode*> children; // значение атрибута -> поддерево
};

// Построение дерева ID3
TreeNode* buildID3(const std::vector<Example>& data,
                   const std::vector<std::string>& attrNames,
                   const std::vector<int>& availableAttributes);

// Классификация нового примера по готовому дереву
std::string classify(const TreeNode* root,
                     const Example& example,
                     const std::vector<std::string>& attrNames);

// Освобождение памяти
void freeTree(TreeNode* root);
