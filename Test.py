# # from Crypto.Cipher import AES
# # from Crypto.Random import get_random_bytes
# # import pandas as pd
# # key = get_random_bytes(16)
# # cipher = AES.new(key, AES.MODE_OCB)
# # nonce = cipher.nonce
# #
# # # data=pd.read_csv("Dataset/rsu1.csv")
# # with open('Dataset/DB1/rsu1.csv', 'rb') as f_in:
# #     data = f_in.read()
# # ciphertext,tag = cipher.encrypt_and_digest(data)
# #
# # with open("data.bin","wb") as f:
# #     f.write(tag)
# #     f.write(nonce)
# #     f.write(ciphertext)
# #
# # with open("data.bin","rb") as f:
# #     tag=f.read(16)
# #     nonce=f.read(15)
# #     ciphertext=f.read()
# # cipher2=AES.new(key, AES.MODE_OCB, nonce=nonce)
# # data2=cipher2.decrypt_and_verify(ciphertext, tag)
# # print(data2)
# from PIL import Image, ImageDraw, ImageFont
#
# # Load your image
# img_path = 'D:\Ajay\Ajay-Electrical\August\Gaurav_Paper2_new\Results\Performance_Plot\During_Harmonics.png'
# output_path = 'D:\Ajay\Ajay-Electrical\August\Gaurav_Paper2_new\Results\Performance_Plot\During_Harmonics_Updated.png'
# img = Image.open(img_path)
#
# # Define where to place the new heading
# title_y = 15  # Vertical position (pixels from top, adjust if needed)
# rect_height = 60  # Height of the rectangle (adjust for your image/font size)
#
# # Create a drawing context
# draw = ImageDraw.Draw(img)
#
# # Mask original title (cover the region with white)
# draw.rectangle([(0, 0), (img.width, rect_height)], fill='white')
#
# # Set font size and type
# try:
#     font = ImageFont.truetype("arial.ttf", 28)  # Use "arial.ttf" if available
# except:
#     font = ImageFont.load_default()  # Fallback to default font
#
# # New heading text
# new_title = "HA-DRL-ANN during Voltage 5th and 7th Order Harmonics"
#
# # Write new title
# draw.text((40, title_y), new_title, fill=(0, 0, 0), font=font)
#
# # Save updated image
# img.save(output_path)
# print("Saved updated image as", output_path)

# import matplotlib.pyplot as plt
#
# models = ["LSTM", "Fed-LSTM", "FSO-Hd-FLTNet"]
# accuracy = [84, 90, 97.6199]
#
# plt.bar(models, accuracy, color=['blue', 'orange', 'green'])
# plt.ylabel("Accuracy (%)")
# # plt.title("Model Accuracy Comparison")
# plt.ylim(0, 100)
# # plt.grid(True, axis='y')
# plt.show()

# import numpy as np
# a=[90,91,64]
# a=np.array(a)
# np.save("Hybrid_drift/Check.npy",
from Proposed_model.proposed_model import proposed_model
from Sub_Functions.Load_data import train_test_splitter1

x_train, x_test, y_train, y_test = train_test_splitter1("DB1", percent=0.7)
result = proposed_model(x_train, x_test, y_train, y_test, 0.7 , "DB1",opt=3)