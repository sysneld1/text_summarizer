# # Hierarchical Text Summarization with LLM
#
# Проект для иерархической суммаризации длинных текстов (например, книг) с использованием локальных LLM
# моделей через llama.cpp. Алгоритм разбивает текст на чанки, создает суммаризации,
# а затем рекурсивно объединяет их в связное повествование.

import os
import re
import time
from llama_cpp import Llama
from typing import List

# Загрузка модели (замени на свой путь к файлу модели)
model_path = r"G:\LLM_models2\Grok-3-reasoning-gemma3-12B-distilled-HF.Q8_0.gguf"
llm = Llama(
    model_path=model_path,
    chat_format="gemma",  # Или попробуй "chatml" для лучшей совместимости
    n_ctx=32768,
    n_threads=8,
    n_gpu_layers=47,
    temperature=0.1,
    max_tokens=8192,
    verbose=True
)


def clean_model_output(text):
    """
    Очистка вывода модели от внутренних рассуждений и служебных тегов.
    """
    # Удаляем блоки <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Удаляем другие возможные теги рассуждений
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    text = re.sub(r'<reflection>.*?</reflection>', '', text, flags=re.DOTALL)
    text = re.sub(r'<scratchpad>.*?</scratchpad>', '', text, flags=re.DOTALL)

    # Удаляем фразы типа "Let me think", "Ok" и т.д.
    thinking_patterns = [
        r'Ok, let me think.*?\n\n',
        r'Let me see.*?\n\n',
        r'Let me figure this out.*?\n\n',
        r'First,.*?\n\n',
        r'I need to.*?\n\n',
        r'So,.*?\n\n',
        r'Alright,.*?\n\n',
        r'Okay,.*?\n\n',
    ]

    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Удаляем лишние пустые строки
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    return text.strip()


def combine_into_narrative(llm, text_list: List[str]) -> str:
    """
    Функция принимает список строк и использует модель LLM для объединения их в одно связное повествование.

    :param llm: Инициализированная модель Llama (например, llm = Llama(...))
    :param text_list: Список строк для объединения
    :return: Один текст, представляющий объединенное повествование
    """
    if not text_list:
        return ""

    system_message = {"role": "system",
                      "content": """Ты — русскоязычный ассистент по обработке текстов.
                                 Используй только русский язык. Используй ТОЛЬКО контекст ниже.
                                 ВАЖНО: Не используй теги <think>, <reasoning> или другие мета-рассуждения.
                                 Просто дай ответ на русском языке."""}

    cont_list = "".join(text_list)
    print(f"\n##########\nСписок после объединения через join \n{cont_list[:500]}...")

    messages = [
        system_message,
        {"role": "user", "content": f"""Соедини следующий список текстов на русском языке из Контекста в единое связное повествование на русском языке.
            Обеспечь, чтобы длина вывода была примерно такой же, как длина объединенных входных текстов.
            НЕ используй теги <think>, <reasoning> или другие мета-рассуждения.
            Просто дай связный текст на русском языке.
            Контекст: \n\n{cont_list}"""
         }
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=8000,
        temperature=0.1,
        stop=["</s>", "Human:", "<think>", "<reasoning>", "<scratchpad>"]
    )

    if response['choices']:
        result = response['choices'][0]['message']['content'].strip()
        # Очищаем вывод от внутренних рассуждений
        result = clean_model_output(result)
        return result
    else:
        return ""


def clean_text(text):
    """Очистка текста от лишних символов и форматирования."""
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'[^\w\s.,!?—–-]', '', text)  # Удаление специальных символов, кроме базовых
    return text.strip()


def chunk_text(text, chunk_size=500, overlap_sentences=3):
    """Разделение текста на чанки по примерно chunk_size символов с нахлестом в overlap_sentences предложений."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    overlap = []

    for sentence in sentences:
        if overlap:
            current_chunk_sentences.extend(overlap)
            current_length += sum(len(s) + 1 for s in overlap)
            overlap = []

        current_chunk_sentences.append(sentence)
        current_length += len(sentence) + 1

        if current_length >= chunk_size:
            chunk_text_str = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text_str)
            if len(current_chunk_sentences) >= overlap_sentences:
                overlap = current_chunk_sentences[-overlap_sentences:]
            else:
                overlap = current_chunk_sentences[:]
            current_chunk_sentences = []
            current_length = 0

    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))

    return chunks


def summarize_chunk(chunk, level=1, summary_file=None):
    """Суммаризация одного чанка с системным сообщением для русского языка."""
    system_message = {"role": "system",
                      "content": """Ты — русскоязычный ассистент. Все ответы давай на русском языке. 
                                 Не используй английский. Используй ТОЛЬКО Фрагмент ниже.
                                 ВАЖНО: Не используй теги <think>, <reasoning> или другие мета-рассуждения.
                                 Не объясняй свои мысли. Просто дай суммаризацию на русском языке."""}

    prompt = f"""Суммируй этот фрагмент текста на русском языке в 5-6 предложениях, включая сюжет, ключевые события, персонажей, диалоги и темы. 
    Фрагмент: {chunk}

    Суммаризация должна быть краткой, но содержать сюжет, ключевые идеи, события и персонажей. 
    Уровень детализации: {level} (1 - самый детальный, выше - более обобщённый). 
    ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ, без введения или заключения. 
    НЕ используй теги <think>, <reasoning> или другие мета-рассуждения.
    Используй только текст из контента ниже."""

    messages = [
        system_message,
        {"role": "user", "content": prompt}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=600,
        temperature=0.1,
        stop=["</s>", "Human:", "<think>", "<reasoning>", "<scratchpad>", "Ok", "So,", "First,"]
    )

    summary = response['choices'][0]['message']['content'].strip()

    # Очищаем вывод от внутренних рассуждений
    summary = clean_model_output(summary)

    # Запись в файл суммаризаций
    if summary_file:
        summary_file.write(f"\n{'=' * 80}\n")
        summary_file.write(f"ЧАНК (уровень {level}):\n")
        summary_file.write(f"{chunk[:500]}...\n\n" if len(chunk) > 500 else f"{chunk}\n\n")
        summary_file.write(f"СУММАРИЗАЦИЯ ЧАНКА:\n")
        summary_file.write(f"{summary}\n")
        summary_file.write(f"{'=' * 80}\n\n")

    # Проверка на английский язык
    english_words = ['the', 'and', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 'as', 'for']
    word_count = len(summary.split())

    if word_count > 10:  # Проверяем только если текст достаточно длинный
        english_word_count = sum(1 for word in english_words if word in summary.lower().split())
        if english_word_count > 2:  # Если найдено больше 2 английских слов
            print("⚠️ Обнаружен английский в сводке чанка. Перегенерирую...")
            if summary_file:
                summary_file.write("⚠️ Обнаружен английский в сводке чанка. Перегенерирую...\n")

            # Более строгий промпт для перегенерации
            strict_prompt = f"""Перепиши эту суммаризацию на русском языке:
            Исходная суммаризация: {summary}

            Перепиши на чистом русском языке без английских слов. 
            Сделай 5-6 предложений о сюжете, событиях и персонажах.
            Только русский язык!"""

            strict_messages = [
                system_message,
                {"role": "user", "content": strict_prompt}
            ]

            response = llm.create_chat_completion(
                messages=strict_messages,
                max_tokens=600,
                temperature=0.3,
                stop=["</s>", "Human:", "<think>"]
            )

            summary = response['choices'][0]['message']['content'].strip()
            summary = clean_model_output(summary)

            if summary_file:
                summary_file.write(f"ИСПРАВЛЕННАЯ СУММАРИЗАЦИЯ:\n{summary}\n\n")

    print(summary[:200] + "..." if len(summary) > 200 else summary)
    return summary


def hierarchical_summarize(summaries, max_group_size=5, level=1, summary_file=None, log_file=None):
    """Иерархическая суммаризация с рекурсией."""
    if log_file:
        log_file.write(f"=== Уровень {level}: Обработка {len(summaries)} суммаризаций ===\n")
        for idx, summ in enumerate(summaries):
            log_file.write(f"Суммаризация {idx + 1}: {summ[:100]}...\n")
        log_file.write("\n")

    if len(summaries) <= 1:
        if log_file:
            log_file.write(f"Уровень {level}: Мало суммаризаций, возвращаем как есть.\n\n")
        return summaries[0] if summaries else ""

    if len(summaries) <= max_group_size:
        combined = "\n\n".join(summaries)

        system_message = {"role": "system",
                          "content": """Ты — русскоязычный ассистент. Все ответы давай на русском языке. 
                                     Не используй английский. 
                                     ВАЖНО: Не используй теги <think>, <reasoning> или другие мета-рассуждения.
                                     Не объясняй свои мысли. Просто дай суммаризацию."""}

        prompt = f"""На основе ТОЛЬКО этих Суммаризаций, создай более обобщённую сводку на русском языке. 
        Суммаризации: {combined}

        Сводка должна объединить сюжет, ключевые идеи, сохраняя последовательность сюжета. 
        Уровень детализации: {level}. 
        ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ, в 5-8 предложениях.
        НЕ используй теги <think>, <reasoning> или другие мета-рассуждения."""

        messages = [
            system_message,
            {"role": "user", "content": prompt}
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=800,
            temperature=0.1,
            stop=["</s>", "Human:", "<think>", "<reasoning>"]
        )

        super_summary = response['choices'][0]['message']['content'].strip()
        super_summary = clean_model_output(super_summary)

        # Запись иерархической суммаризации в файл
        if summary_file:
            summary_file.write(f"\n{'#' * 80}\n")
            summary_file.write(f"ИЕРАРХИЧЕСКАЯ СУММАРИЗАЦИЯ (уровень {level}):\n")
            summary_file.write(f"Количество исходных суммаризаций: {len(summaries)}\n")
            summary_file.write(f"Результат:\n{super_summary}\n")
            summary_file.write(f"{'#' * 80}\n\n")

        if log_file:
            log_file.write(
                f"Уровень {level}: Суммирована группа из {len(summaries)} в супер-суммаризацию: {super_summary[:200]}...\n\n")

        print(f"Уровень {level}: {super_summary[:200]}...")
        return super_summary

    # Рекурсия: разделяем на группы
    groups = [summaries[i:i + max_group_size] for i in range(0, len(summaries), max_group_size)]

    if log_file:
        log_file.write(f"\nУровень {level}: Разделено на {len(groups)} групп по {max_group_size}.\n")
        for g_idx, group in enumerate(groups):
            log_file.write(f"  Группа {g_idx + 1}: {len(group)} суммаризаций\n")
        log_file.write("\n")

    super_summaries = []

    for group in groups:
        super_summary = hierarchical_summarize(group, max_group_size, level + 1, summary_file, log_file)
        super_summaries.append(super_summary)

        if log_file:
            log_file.write(f"Уровень {level}: Добавлена супер-суммаризация группы: {super_summary[:200]}...\n")
            log_file.write(f"\nИтого по группе super_summaries: {len(super_summaries)} элементов\n")

        print(f"Добавлена супер-суммаризация: {super_summary[:100]}...")

    # Объединение супер-суммаризаций в повествование
    comb = combine_into_narrative(llm, super_summaries)

    if summary_file:
        summary_file.write(f"\n{'@' * 80}\n")
        summary_file.write(f"ОБЪЕДИНЕНИЕ СУПЕР-СУММАРИЗАЦИЙ (уровень {level}):\n")
        summary_file.write(f"Исходные супер-суммаризации: {len(super_summaries)}\n")
        for i, summ in enumerate(super_summaries):
            summary_file.write(f"\nСупер-суммаризация {i + 1}:\n{summ[:300]}...\n" if len(
                summ) > 300 else f"\nСупер-суммаризация {i + 1}:\n{summ}\n")
        summary_file.write(f"\nОбработанный список (объединенное повествование):\n{comb}\n")
        summary_file.write(f"{'@' * 80}\n\n")

    # Рекурсивно суммаризируем супер-суммаризации
    final = hierarchical_summarize(super_summaries, max_group_size, level, summary_file, log_file)
    final = clean_model_output(final)

    if log_file:
        log_file.write(f"Уровень {level}: Финальная рекурсивная суммаризация супер-групп завершена.\n\n")
        log_file.write(f"Финальная суммаризация: {final[:500]}...\n")

    return final


def main():
    # Начало измерения времени
    start_time = time.time()

    # Создаем файлы для записи результатов
    summary_output_file = "Summary_Detailed.txt"
    final_output_file = "Output_summary.txt"
    log_file_path = "hierarchical_log.txt"

    # Открываем файлы для записи
    summary_file = open(summary_output_file, 'w', encoding='utf-8')
    final_file = open(final_output_file, 'w', encoding='utf-8')
    log_file = open(log_file_path, "w", encoding="utf-8")

    # Заголовки файлов
    summary_file.write("=" * 100 + "\n")
    summary_file.write("ПОДРОБНАЯ СУММАРИЗАЦИЯ ТЕКСТА\n")
    summary_file.write("=" * 100 + "\n\n")

    final_file.write("=" * 100 + "\n")
    final_file.write("ИТОГОВАЯ СУММАРИЗАЦИЯ ТЕКСТА\n")
    final_file.write("ВСЕ РАССУЖДЕНИЯ И ТЕГИ УДАЛЕНЫ\n")
    final_file.write("=" * 100 + "\n\n")

    log_file.write("Лог иерархической суммаризации\n\n")

    # Чтение файла с книгой
    book_file = r"G:\books\Master_i_Margarita.txt"

    if not os.path.exists(book_file):
        print(f"❌ Файл {book_file} не найден!")
        summary_file.close()
        final_file.close()
        log_file.close()
        return

    try:
        with open(book_file, 'r', encoding='cp1251') as book_f:
            full_text = book_f.read()
    except UnicodeDecodeError:
        with open(book_file, 'r', encoding='utf-8') as book_f:
            full_text = book_f.read()

    full_text = clean_text(full_text)
    print(f"📖 Текст загружен: {len(full_text)} символов.")
    summary_file.write(f"📖 Исходный текст загружен: {len(full_text)} символов.\n\n")

    # Разделение на чанки
    chunks = chunk_text(full_text, chunk_size=3000)
    print(f"🔢 Текст разделён на {len(chunks)} чанков.")
    summary_file.write(f"🔢 Текст разделён на {len(chunks)} чанков.\n\n")

    # Суммаризация чанков
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Суммаризирую чанк {i + 1}/{len(chunks)}...")
        summary_file.write(f"\n{'=' * 80}\n")
        summary_file.write(f"ОБРАБОТКА ЧАНКА {i + 1} из {len(chunks)}\n")
        summary_file.write(f"{'=' * 80}\n")

        summary = summarize_chunk(chunk, level=1, summary_file=summary_file)
        summaries.append(summary)
        print(f"✅ Чанк {i + 1} готов: {len(summary)} символов.")

    # Иерархическая суммаризация
    print("🏗️ Начинаю иерархическую суммаризацию...")
    summary_file.write("\n\n" + "=" * 100 + "\n")
    summary_file.write("НАЧАЛО ИЕРАРХИЧЕСКОЙ СУММАРИЗАЦИИ\n")
    summary_file.write("=" * 100 + "\n\n")

    final_summary = hierarchical_summarize(summaries, summary_file=summary_file, log_file=log_file)

    # Финальная общая сводка
    print("📝 Создаю финальную общую сводку...")
    system_message = {"role": "system",
                      "content": """Ты — русскоязычный ассистент. Все ответы давай на русском языке. 
                                 Не используй английский. 
                                 ВАЖНО: Не используй теги <think>, <reasoning> или другие мета-рассуждения.
                                 Не объясняй свои мысли. Просто дай суммаризацию."""}

    final_prompt = f"""На основе ТОЛЬКО ЭТОЙ иерархической сводки, создай полную общую сводку в 10-20 предложениях на русском языке. 
    Иерархическая сводка: {final_summary}

    Сводка должна охватывать сюжет, основные темы, ключевых персонажей и сюжетные повороты. 
    Из общей сводки должны быть понятны сюжет, сюжетные повороты и ключевые персонажи. 
    ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
    НЕ используй теги <think>, <reasoning> или другие мета-рассуждения."""

    messages = [
        system_message,
        {"role": "user", "content": final_prompt}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1600,
        temperature=0.1,
        stop=["</s>", "Human:", "<think>", "<reasoning>", "Ok,", "So,", "First,"]
    )

    overall_summary = response['choices'][0]['message']['content'].strip()
    overall_summary = clean_model_output(overall_summary)

    # Запись результатов в final_file (итоговый файл)
    final_file.write("=== ОБЩАЯ СВОДКА (ОЧИЩЕНА ОТ РАССУЖДЕНИЙ) ===\n\n")
    final_file.write(overall_summary)
    final_file.write("\n\n" + "=" * 80 + "\n\n")

    final_file.write("=== ИЕРАРХИЧЕСКАЯ СВОДКА (ОЧИЩЕНА ОТ РАССУЖДЕНИЙ) ===\n\n")
    final_file.write(final_summary)
    final_file.write("\n\n" + "=" * 80 + "\n\n")

    final_file.write("=== ПРЕДВАРИТЕЛЬНЫЕ СУММАРИЗАЦИИ ЧАНКОВ ===\n")
    for i, summ in enumerate(summaries):
        final_file.write(f"\n{'=' * 60}\n")
        final_file.write(f"Чанк {i + 1}:\n")
        cleaned_summ = clean_model_output(summ)
        final_file.write(f"{cleaned_summ}\n")

    # Запись результатов в summary_file (подробный файл)
    summary_file.write("\n\n" + "*" * 100 + "\n")
    summary_file.write("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ\n")
    summary_file.write("*" * 100 + "\n\n")

    summary_file.write("=== ИТОГОВАЯ ОБЩАЯ СВОДКА (ОЧИЩЕНА) ===\n\n")
    summary_file.write(overall_summary)
    summary_file.write("\n\n" + "=" * 80 + "\n\n")

    summary_file.write("=== ФИНАЛЬНАЯ ИЕРАРХИЧЕСКАЯ СВОДКА (ОЧИЩЕНА) ===\n\n")
    summary_file.write(final_summary)

    print(f"🎉 Сводка сохранена в {final_output_file}")
    print(f"📋 Подробная суммаризация сохранена в {summary_output_file}")
    print(f"📝 Лог сохранен в {log_file_path}")

    print("\n--- Превью финальной сводки ---")
    print(overall_summary[:500] + "..." if len(overall_summary) > 500 else overall_summary)

    # Конец измерения времени и вывод
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_hours = elapsed / 3600

    print(f"\nВремя исполнения программы: {elapsed:.2f} секунд ({elapsed_hours:.2f} часов).")

    # Запись времени в файлы
    time_info = f"\n\nВремя исполнения программы: {elapsed:.2f} секунд ({elapsed_hours:.2f} часов)."
    final_file.write(time_info)
    summary_file.write(time_info)
    log_file.write(time_info)

    # Закрытие файлов
    summary_file.close()
    final_file.close()
    log_file.close()


if __name__ == "__main__":
    main()