#pragma once

#include "id3.h"

#include <string>

// Красивый вывод дерева в консоль
void printTree(const TreeNode* node,
               const std::string& prefix = "",
               bool isLast = true);
