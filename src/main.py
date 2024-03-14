# Web demo การทำงานของระบบ
import gradio as gr
from Chatbot import Chatbot

bot = Chatbot()
def chat_interface(question, history):
    response = bot.predict(question)
    answer = response["answer"]
    return answer


if __name__ == "__main__":
    EXAMPLE = ["หลิน ไห่เฟิง มีชื่อเรียกอีกชื่อว่าอะไร" , "ใครเป็นผู้ตั้งสภาเศรษฐกิจโลกขึ้นในปี พ.ศ. 2514 โดยทุกปีจะมีการประชุมที่ประเทศสวิตเซอร์แลนด์", "โปรดิวเซอร์ของอัลบั้มตลอดกาล ของวงคีรีบูนคือใคร", "สกุลเดิมของหม่อมครูนุ่ม นวรัตน ณ อยุธยา คืออะไร"]
    interface = gr.ChatInterface(fn=chat_interface, examples=EXAMPLE)
    interface.launch()