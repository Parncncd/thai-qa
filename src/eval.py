from Chatbot import Chatbot

bot = Chatbot()
models = [
    "wangchanberta",
    # "mdeberta",
]
result = bot.eval(model_name='wangchanberta')
# result = bot.eval(model_name='wangchanberta',answer_with_model=True)
print(result)