import random

# 1~30 badmood
# 35~49 calm
file_names = [f"{i}.mp4" for i in range(1, 46)]
# file_names_2 = [f"{i}.mp4" for i in range(35, 50)]
# file_names.extend(file_names_2)
random.shuffle(file_names)
# with open(r'C:\\junior_project\\experiment2_r1\\random_video.txt', 'w+') as file:
#     for name in file_names:
#         file.write("static\\videos\\" +name + "\n")
print(len(file_names))
chunk1 = file_names[:15]
chunk2 = file_names[15:30]
chunk3 = file_names[30:]
print(len(chunk1))
print(len(chunk2))
print(len(chunk3))
with open(r'C:\\junior_project\\experiment2_r1\\random_video_1.txt', 'w+') as file:
    for name in chunk3:
        file.write("static\\videos\\" +name + "\n")

file.close()
print("done")

with open(r'C:\\junior_project\\experiment2_r2\\random_video_2.txt', 'w+') as file:
    for name in chunk1:
        file.write("static\\videos\\" +name + "\n")

file.close()
print("done")
with open(r'C:\\junior_project\\experiment2_r3\\random_video_3.txt', 'w+') as file:
    for name in chunk2:
        file.write("static\\videos\\" +name + "\n")

file.close()
print("done")
