import random

# 1~30 badmood
# 35~49 calm
file_names = [f"{i}.mp4" for i in range(1, 31)]
file_names_2 = [f"{i}.mp4" for i in range(35, 50)]
file_names.extend(file_names_2)
random.shuffle(file_names)
with open(r'/home/baisp2/junior_project/experiment_1/random_video.txt', 'w+') as file:
    for name in file_names:
        file.write("/home/baisp2/junior_project/experiment_1/"+name + "\n")
file.close()
print("done")
