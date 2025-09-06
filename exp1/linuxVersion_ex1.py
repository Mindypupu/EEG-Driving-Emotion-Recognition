import random
import subprocess
import tkinter as tk
from tkinter import messagebox
import time
import csv
import os

# CSV 文件設定
csv_filename = "data.csv"
if not os.path.exists(csv_filename):  # 如果檔案不存在，新增標題行
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video num", "valance", "arousal"])

video_files = []
with open("random_video.txt", "r") as file:
    for line in file:
        video_files.append(line.strip())

remaining_videos = video_files.copy()
root = tk.Tk()

# 設定視窗大小
window_width, window_height = 1440, 900
screen_width, screen_height = 1920, 1080

pos_x = (screen_width - window_width) // 2
pos_y = (screen_height - window_height) // 2
root.geometry(f'{window_width}x{window_height}+{pos_x}+{pos_y}')  # 簡單設定視窗位置
root.title('Experiment')
root.configure(bg="#5B5B5B")

def play_video(video_path):
    """使用 MPV 播放影片（Linux 適用）"""
    try:
        subprocess.run(["mpv", "--geometry=1280*720+320+180",  # 固定視窗大小並置中
            "--no-border",  # 隱藏視窗框架
            "--autofit=1280x720",  # 設定視窗最大為 1280x720
            "--vf=scale=1280:720",  # 強制縮放影片至 1280x720
            "--fullscreen=no",  # 確保不是全螢幕
            "--no-keepaspect",  # 取消保持原始長寬比
            video_path])
            
        #time.sleep(1)
        #subprocess.run(["wmctrl", "-r", "mpv", "-e", "0,320,180,-1,-1"])
    except FileNotFoundError:
        print("MPV 播放器未安裝，請使用 `sudo apt install mpv` 安裝")
        return


def countdown_timer(seconds, callback):
    """倒數計時畫面，倒數結束後執行 callback"""
    for widget in root.winfo_children():
        widget.destroy()  # 清除現有元件
    
    countdown_window = tk.Toplevel(root)
    countdown_window.title("Countdown")
    countdown_window.geometry("600x300+500+300")
    countdown_window.configure(bg="#5B5B5B")

    countdown_label = tk.Label(countdown_window, text="", bg="#5B5B5B", fg="white", font=("song ti", 50, "bold"))
    countdown_label.pack(pady=40)
    
    skip_label = tk.Label(countdown_window, text="或按X跳過", bg="#5B5B5B", fg="white", font=("mincho", 20, "bold"))
    skip_label.pack()
    
    def skip_countdown(event=None):
    	countdown_window.destroy()
    	callback()
    
    countdown_window.bind("<space>", skip_countdown)	
    
    def update_timer():
        nonlocal seconds
        if seconds > 0:
            countdown_label.config(text=f"倒數: {seconds} 秒")
            seconds -= 1
            root.after(1000, update_timer)
        else:
            skip_countdown()

    update_timer()


def show_affect_grid():
    """顯示情感標示網格"""
    for widget in root.winfo_children():
        widget.destroy()

    question = tk.Label(root, text='請標示你現在的心情', bg='#5B5B5B', fg="#FFFFFF",
                        font=("song ti", 35, "bold"))
    arousal = tk.Label(root, text='喚醒', bg='#5B5B5B', fg="#FFFFFF",
                       font=("song ti", 20, "bold"))
    sleep = tk.Label(root, text='想睡', bg='#5B5B5B', fg="#FFFFFF",
                     font=("song ti", 20, "bold"))
    pleasant = tk.Label(root, text='愉快', bg='#5B5B5B', fg="#FFFFFF",
                        font=("song ti", 20, "bold"))
    unpleasant = tk.Label(root, text='不愉快', bg='#5B5B5B', fg="#FFFFFF",
                          font=("song ti", 20, "bold"))
    canvas = tk.Canvas(root, width=450, height=450)
    question.place(x=505, y=30)
    arousal.place(x=695, y=120)
    unpleasant.place(x=380, y=380)
    canvas.place(x=500, y=170)
    pleasant.place(x=980, y=380)
    sleep.place(x=695, y=640)

    cell_size = 50
    for i in range(9):
        for j in range(9):
            x1 = j * cell_size
            y1 = i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            canvas.create_rectangle(x1, y1, x2, y2, outline="black")

    rect_size = 50
    x, y = 200, 200
    rect = canvas.create_rectangle(x, y, x + rect_size, y + rect_size, fill="red")
    move_step = 50

    def move_rect(event):
        nonlocal x, y
        
        if event.keysym == 'Up' and y > 0:
            canvas.move(rect, 0, -move_step)
            y -= move_step
        elif event.keysym == 'Down' and y < 400:
            canvas.move(rect, 0, move_step)
            y += move_step
        elif event.keysym == 'Left' and x > 0:
            canvas.move(rect, -move_step, 0)
            x -= move_step
        elif event.keysym == 'Right' and x < 400:
            canvas.move(rect, move_step, 0)
            x += move_step
        elif event.keysym == 'Return':
            video_name = random_video.split(".mp4")[0]
            x_val = (x - 200) / 50
            y_val = (y - 200) / 50
        
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([video_name, x_val, y_val])

            # 每播放五部影片且提交後，倒數 30 秒
            if (len(video_files) - len(remaining_videos)) % 5 == 0:
                countdown_timer(30, start_experiment)
            else:
                start_experiment()
  
    # 綁定按鍵事件
    root.bind("<KeyPress-Up>", move_rect)
    root.bind("<KeyPress-Down>", move_rect)
    root.bind("<KeyPress-Left>", move_rect)
    root.bind("<KeyPress-Right>", move_rect)
    root.bind("<KeyPress-Return>", move_rect)

def start_experiment():
    """開始播放影片並進行實驗"""
    if remaining_videos:
        global random_video
        random_video = remaining_videos.pop(0)
        video_path = random_video
       # video_path = f"/home/baisp2/junior_project/experiment_1/{random_video}"
       # video_thread = threading.Thread(target=lambda: play_video(video_path))
       # video_thread.start()
       # video_thread.join()
        subprocess.run(["mpv", video_path])
        print("play video") 

        show_affect_grid()
        print("end show")
    else:
        # 實驗結束，儲存 CSV 檔案
        #with open(csv_filename, "a", newline="") as f:
        #    writer = csv.writer(f)
        #    writer.writerows(csv_data)
        
        messagebox.showinfo("完成", "實驗結束")
        root.quit()


def show_start_page():
    """顯示實驗起始畫面"""
    for widget in root.winfo_children():
        widget.destroy()

    start_label = tk.Label(root, text="歡迎來到實驗！", bg='#5B5B5B', fg="#FFFFFF",
                           font=("song ti", 50, "bold"))
    start_label.place(x=500, y=300)

    start_button = tk.Button(root, text="開始", font=("mincho", 30), command=start_experiment)
    start_button.place(x=650, y=500)


show_start_page()

root.mainloop()

