from flask import Flask, render_template, jsonify, request
import csv
import os
from psychopy import parallel
from time import sleep

add = 0X3Ff8
pport = parallel.ParallelPort(address=add)

app = Flask(__name__)

file_path = 'C:\\junior_project\\experiment2_r3\\video_data_3.csv'
video_data_dict = {}

video_files = []
with open("C:\\junior_project\\experiment2_r3\\random_video_3.txt", 'r') as file:
    for line in file:
        video_files.append(line.strip())
print(video_files)
# video_files = [
#     "static/videos/1.mp4",
#     "static/videos/2.mp4",
#     "static/videos/3.mp4",
#     "static/videos/4.mp4",
#     "static/videos/5.mp4",
#     "static/videos/6.mp4",
#     "static/videos/7.mp4",
#     "static/videos/8.mp4",
#     "static/videos/9.mp4",
#     "static/videos/10.mp4",
#     # 更多影片路徑...
# ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos', methods=['GET'])
def get_videos():
    return jsonify(video_files)

@app.route('/submit', methods=['POST'])
def submit_data():
    data = request.get_json()
    print(data) 
    with open(file_path, mode='a', newline='') as file:
        fieldnames = ['videoName', 'x', 'y', 'currentTime', 'startTime', 'endTime']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if file.tell() == 0:
            writer.writeheader()

        row = {
            'videoName': data['videoName'],
            'x': data.get('x', ''),  
            'y': data.get('y', ''),  
            'currentTime': data.get('currentTime', ''), 
            'startTime': data.get('startTime', ''),  
            'endTime': data.get('endTime', '')  
        }
        writer.writerow(row)

    return 'Success', 200


@app.route('/trigger', methods=['POST'])
def trigger():
    data = request.json
    code = data.get("code", 1)
    print(f"Trigger code received: {code}")

    pport.setData(64)

    print("signal sent")

    sleep(0.01)

    pport.setData(0)

    print("signal clear")


    return 'Triggered!', 200

if __name__ == '__main__':
    app.run(debug=True)
