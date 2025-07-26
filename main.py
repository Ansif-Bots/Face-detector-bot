import telebot
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
import os
from telebot import types

# Bot token from environment
BOT_TOKEN = os.environ.get("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

# Mediapipe face detection
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "👋🏻 Welcome to *Face Crop Bot!* \n\n📸 Send me a photo and I’ll give you a *transparent head-only PNG!*", parse_mode="Markdown")

# Photo handler
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "⏳ Detecting face and cropping... Please wait 😅")
    
    # Download image
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    nparr = np.frombuffer(downloaded_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    
    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            ww = int(box.width * w)
            hh = int(box.height * h)
            
            # Padding to include hair (head only)
            pad = 50
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + ww + pad)
            y2 = min(h, y + hh + pad)
            
            cropped = img[y1:y2, x1:x2]
            
            # Remove background
            removed_bg = remove(cropped)
            
            output_path = "head.png"
            with open(output_path, "wb") as f:
                f.write(removed_bg)
            
            with open(output_path, 'rb') as photo:
                bot.send_document(message.chat.id, photo, caption="🎉 Here is your head-only PNG!")

            return

    bot.reply_to(message, "❌ Face not detected properly. Please try again with a clearer photo.")

# Optional UptimeRobot ping
@bot.message_handler(commands=['ping'])
def ping(message):
    bot.reply_to(message, "🏓 I'm alive and running!")

# Start bot
print("🤖 Bot started...")
bot.infinity_polling()
