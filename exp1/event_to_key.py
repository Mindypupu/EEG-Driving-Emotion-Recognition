import evdev
from pynput.keyboard import Controller, Key

device_path = "/dev/input/event14"  # 設備路徑
dev = evdev.InputDevice(device_path)
keyboard = Controller()

# 🔹 設定對應的鍵盤方向鍵
key_map = {
    17: {1: Key.down, -1: Key.up},  # 17: 上下（1=下，-1=上）
    16: {1: Key.right, -1: Key.left},  # 16: 左右（1=右，-1=左）
    290: Key.enter,  # 290: Enter
    288: Key.space   # 288: 空白鍵
}

print("🔹 監聽設備輸入中...（按 Ctrl+C 結束）")
for event in dev.read_loop():
    if event.type in [evdev.ecodes.EV_KEY, evdev.ecodes.EV_ABS] and event.value != 0:  # 只處理「按下」事件
        if event.code in key_map:
            mapped_key = key_map[event.code]
            if isinstance(mapped_key, dict):  # 針對上下左右的變數值
                mapped_key = mapped_key.get(event.value, None)
            if mapped_key:
                print(f"🔹 轉換: event {event.code} → 按鍵 {mapped_key}")
                keyboard.press(mapped_key)
                keyboard.release(mapped_key)  # 立即釋放，模擬單次按鍵
