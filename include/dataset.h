#pragma once

#include <string>
#include <vector>

// Один пример обучающей выборки: атрибуты + целевая метка
struct Example {
    std::vector<std::string> attrs;
    std::string label;
};

// Имена атрибутов (всего 4)
std::vector<std::string> getAttributeNames();

// Обучающая выборка из 14 примеров (задача выбора поставщика)
std::vector<Example> getTrainingData();

// Сохранение выборки в CSV-файл в папку data/
void saveDatasetToCSV(const std::string& filename,
                      const std::vector<Example>& data,
                      const std::vector<std::string>& attrNames);
