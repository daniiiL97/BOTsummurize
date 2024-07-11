import time
import telebot
from telebot import types
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import threading
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv
import os

load_dotenv()
KEY = os.getenv('BOTKEY')
bot = telebot.TeleBot(KEY)

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
user_summary_length = {}

@bot.message_handler(commands=['start'])
def start(message):
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Начать", callback_data="start_button"))
    bot.send_message(message.chat.id, "Добро пожаловать в бота для суммаризации текста! 📚✨\n"
                                      "Этот бот использует модель FRED-T5-Summarizer для создания кратких суммаризаций текстов на русском языке.",
                                      reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == "start_button")
def next(call):
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Далее", callback_data="next"))
    bot.send_message(call.message.chat.id,
                     "Как использовать бота:\n"
                     "1. Начало работы: Нажмите кнопку 'Начать' ниже. ▶️\n"
                     "2. Выбор длины суммаризации: После нажатия кнопки 'Начать' выберите длину суммаризации:\n"
                     "   - Краткая: Для самых кратких обзоров (примерно 20 слов). ✨\n"
                     "   - Средняя: Для более детализированных суммаризаций (примерно 50 слов). 🌟\n"
                     "   - Подробная: Для максимально подробных суммаризаций (примерно 100 слов). 🌠\n"
                     "3. Ввод текста: После выбора длины суммаризации введите текст, который хотите суммировать. 🖊️\n"
                     "4. Получение результата: Подождите, пока бот обработает ваш текст и создаст суммаризацию. ⏳\n\n"
                     "Примечание: Убедитесь, что ваш текст содержит не более 600 слов для корректной работы модели. ⚠️\n\n"
                     "Начните с нажатия кнопки 'Далее' и следуйте инструкциям. Приятного использования! 🎉",
                     reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == "next")
def start_summary_process(call):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Краткая")
    btn2 = types.KeyboardButton("Средняя")
    btn3 = types.KeyboardButton("Подробная")
    markup.add(btn1, btn2, btn3)
    bot.send_message(call.message.chat.id, 'Привет! Выбери размер суммаризации: Краткая, Средняя или Подробная.',
                     reply_markup=markup)

@bot.message_handler(func=lambda message: message.text in ["Краткая", "Средняя", "Подробная"])
def set_summary_length(message):
    if message.text == "Краткая":
        user_summary_length[message.chat.id] = 20
    elif message.text == "Средняя":
        user_summary_length[message.chat.id] = 50
    elif message.text == "Подробная":
        user_summary_length[message.chat.id] = 100
    bot.send_message(message.chat.id,
                     f'Выбран размер суммаризации: {message.text}. Теперь введи текст для суммаризации.',
                     reply_markup=types.ReplyKeyboardRemove())

@bot.message_handler(content_types=['text'])
def answer(message):
    if message.chat.id not in user_summary_length:
        bot.send_message(message.chat.id,
                         'Пожалуйста, сначала выбери размер суммаризации, используя команды: Краткая, Средняя или Подробная.')
        return

    bot.send_message(message.chat.id, 'Анализируем текст...')
    article_text = message.text
    input_ids = tokenizer(
        [article_text],
        max_length=512,  
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    summary_length = user_summary_length[message.chat.id]

    def generate_summary():
        nonlocal input_ids, summary_length, message
        progress_message = bot.send_message(message.chat.id, 'Генерация суммаризации...')
        for i in tqdm(range(10), desc="Генерация"):
            time.sleep(1)
            bot.edit_message_text(chat_id=message.chat.id, message_id=progress_message.message_id,
                                  text=f'Генерация суммаризации... {i * 10}%')

        output_ids = model.generate(
            input_ids=input_ids,
            min_length=summary_length,
            max_length=512,
            no_repeat_ngram_size=2
        )[0]

        summary = tokenizer.decode(output_ids, skip_special_tokens=True)

        bot.send_message(message.chat.id, summary)
        do_again(message)

    threading.Thread(target=generate_summary).start()

def do_again(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Краткая")
    btn2 = types.KeyboardButton("Средняя")
    btn3 = types.KeyboardButton("Подробная")
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.chat.id,
                     'Хочешь проверить другой текст? Выбери размер суммаризации: Краткая, Средняя или Подробная.',
                     reply_markup=markup)

if __name__ == '__main__':
    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(5)
