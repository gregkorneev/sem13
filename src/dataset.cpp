#include "dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>

std::vector<std::string> getAttributeNames() {
    return {
        "Цена",
        "Качество",
        "Срок поставки",
        "Надёжность"
    };
}

std::vector<Example> getTrainingData() {
    // 14 примеров, 4 атрибута, целевой атрибут — label ("Да"/"Нет")
    return {
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
}

void saveDatasetToCSV(const std::string& filename,
                      const std::vector<Example>& data,
                      const std::vector<std::string>& attrNames) {
    namespace fs = std::filesystem;

    fs::path path(filename);
    fs::path dir = path.parent_path();
    if (!dir.empty()) {
        fs::create_directories(dir);
    }

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Не удалось открыть файл для записи: " << filename << '\n';
        return;
    }

    // ===== ЗАГОЛОВОК =====
    // Цена;Качество;Срок поставки;Надёжность;Решение
    for (std::size_t i = 0; i < attrNames.size(); ++i) {
        out << attrNames[i];
        if (i + 1 < attrNames.size()) {
            out << ';'; // разделитель колонок — ТОЧКА С ЗАПЯТОЙ
        }
    }
    out << ";Решение\n";

    // ===== СТРОКИ ДАННЫХ =====
    for (const auto& ex : data) {
        for (std::size_t i = 0; i < ex.attrs.size(); ++i) {
            out << ex.attrs[i];
            if (i + 1 < ex.attrs.size()) {
                out << ';';
            }
        }
        out << ';' << ex.label << '\n';
    }

    std::cout << "Таблица обучающей выборки сохранена в CSV: "
              << filename << '\n';
}
