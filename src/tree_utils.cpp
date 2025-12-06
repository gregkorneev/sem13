#include "tree_utils.h"

#include <iostream>

void printTree(const TreeNode* node,
               const std::string& prefix,
               bool isLast) {
    if (!node) return;

    std::cout << prefix;
    std::cout << (isLast ? "└── " : "├── ");

    if (node->isLeaf) {
        std::cout << "[КЛАСС: " << node->label << "]\n";
    } else {
        std::cout << "[АТРИБУТ: " << node->label << "]\n";
    }

    // Дети
    const std::size_t childCount = node->children.size();
    std::size_t i = 0;
    for (const auto& [value, child] : node->children) {
        bool childIsLast = (++i == childCount);

        std::cout << prefix
                  << (isLast ? "    " : "│   ")
                  << "(" << value << ") "
                  << (childIsLast ? "" : "") ;

        // Чтобы ветка с подписью значения не "съедала" начало строки,
        // мы просто вызываем printTree с обновлённым префиксом.
        printTree(child,
                  prefix + (isLast ? "    " : "│   "),
                  childIsLast);
    }
}
