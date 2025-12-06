#include <iostream>
#include <vector>

#include "dataset.h"
#include "id3.h"
#include "tree_utils.h"

int main() {
    // 1. Подготовка данных
    auto attrNames = getAttributeNames();
    auto data      = getTrainingData();

    std::cout << "Количество примеров: " << data.size() << '\n';

    // 2. Сохранение таблицы предметной области в CSV
    saveDatasetToCSV("data/supplier_dataset.csv", data, attrNames);

    // 3. Построение дерева ID3
    std::vector<int> availableAttributes;
    availableAttributes.reserve(attrNames.size());
    for (int i = 0; i < static_cast<int>(attrNames.size()); ++i) {
        availableAttributes.push_back(i);
    }

    TreeNode* root = buildID3(data, attrNames, availableAttributes);

    std::cout << "\nДерево решений (алгоритм ID3) для задачи выбора поставщика:\n";
    printTree(root);

    // 4. Пример классификации нового поставщика
    Example newSupplier{
        /* attrs: */ {"средняя", "высокое", "быстрая", "высокая"},
        /* label: */ ""
    };

    std::string decision = classify(root, newSupplier, attrNames);

    std::cout << "\nКлассификация нового поставщика "
              << "(Цена=средняя, Качество=высокое, Срок=быстрая, Надёжность=высокая): "
              << decision << '\n';

    // 5. Освобождение памяти
    freeTree(root);

    return 0;
}
