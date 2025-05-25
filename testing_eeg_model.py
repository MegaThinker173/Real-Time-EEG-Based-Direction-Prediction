# When running this program in "working_with_eeg_models":
# source venv/bin/activate

import csv
import random
import time
from datetime import datetime
import pygame
import asyncio
import json
import ssl
import websockets
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
reset = False
from preprocess import pre_process_item
import pickle

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DOT_RADIUS = 10
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
DIRECTIONS = ['down', 'right', 'left', 'up']
# original: ['up', 'down', 'left', 'right']
# label encoder on confusion matrix: ['down', 'left', 'right', 'up'] (alphabet order)
# train and test loop: ['right', 'up', 'down', 'left']

WINDOW_SIZE = 800
BUFFER = deque(maxlen=WINDOW_SIZE)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

columns = [
    "COUNTER",
    "INTERPOLATED",
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    "RAW_CQ",
    "MARKER_HARDWARE",
    "MARKERS"
]

# Load model
MODEL_PATH = "/Users/prestonbadger/Downloads/may25_mrquy_eeg_prediction_model.h5"
model = load_model(MODEL_PATH)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dot Movement Game")
font = pygame.font.Font(None, 36)

# Game state
center_dot = [WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]
baseline_time = time.time()
baseline_active = True
game_started = False
start_time = None
actions = []
baseline_mean = None

# Load label encoder to determine correct direction order
with open("/Users/prestonbadger/Downloads/may25_mrquy_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set DIRECTIONS based on the trained model's label order
DIRECTIONS = ['down', 'right', 'left', 'up']
# (list(label_encoder.classes_))
# print(f"Loaded DIRECTIONS from label_encoder: {DIRECTIONS}")

DIRECTION_TO_POSITION = {
    'down':  (WINDOW_WIDTH // 2, WINDOW_HEIGHT - DOT_RADIUS),
    'right': (WINDOW_WIDTH - DOT_RADIUS, WINDOW_HEIGHT // 2),
    'left':  (DOT_RADIUS, WINDOW_HEIGHT // 2),
    'up':    (WINDOW_WIDTH // 2, DOT_RADIUS)
}
full_sides = [DIRECTION_TO_POSITION[dir] for dir in DIRECTIONS]

# Reverse lookup: position -> direction
POSITION_TO_DIRECTION = {pos: direction for direction, pos in zip(DIRECTIONS, full_sides)}
'''
Top: (WINDOW_WIDTH // 2, DOT_RADIUS) -> Up
Bottom: (WINDOW_WIDTH // 2, WINDOW_HEIGHT - DOT_RADIUS) -> Down
Left: (DOT_RADIUS, WINDOW_HEIGHT // 2) -> Left
Right: (WINDOW_WIDTH - DOT_RADIUS, WINDOW_HEIGHT // 2) -> Right
'''

def get_remaining_side_and_action(sides):
    for side in full_sides:
        if side not in sides:
            direction = POSITION_TO_DIRECTION[side]
            return direction, side

def generate_new_side_dots(sides):
    return random.sample(sides, 3)

def move_ball(directionuseristhinking, correct_direction):
    if directionuseristhinking == correct_direction:
        print("✅ Prediction correct")
        print(f"Correct direction: {correct_direction}")
        print(f"Direction user is thinking: {directionuseristhinking}")
        print(f"⭐ Ball moved {correct_direction}!")
        global reset
        reset = True
    else:
        print("❌ Prediction wrong")
        print(f"Direction user is thinking: {directionuseristhinking}")
        print(f"Correct Direction: {correct_direction}")
    time.sleep(2)

def process_baseline(segment):
    global baseline_mean
    if baseline_mean is None:
        baseline_array = np.array(segment)
        baseline_mean = np.mean(baseline_array, axis=0)
        print("baseline_mean", baseline_mean)

def predict_and_move(segment):
    global current_action
    if len(segment) != 800 or baseline_mean is None:
        return
    print("np.array(segment).shape", np.array(segment).shape)
    # segment = np.array(segment).reshape(1, 800, 14)
    print(baseline_mean)
    segment = np.array(segment) - baseline_mean

    data = pre_process_item(np.array(segment))
    preds = model.predict(data)[0]
    pred_idx = np.argmax(preds)
    pred_label = DIRECTIONS[pred_idx]
    print(f"pred_idx: {pred_idx}")

    direction_user_is_thinking = pred_label
    print("Prediction: ", {d: round(p, 3) for d, p in zip(DIRECTIONS, preds)}) # Just to confirm that direction_user_is_thinking matches the direction with highest probability (direction that user is thinking)
    move_ball(direction_user_is_thinking, current_action) # current_action = correct direction that the ball should move

async def get_eeg_input():
    global BUFFER
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect("wss://localhost:6868", ssl=ssl_context) as ws:
        async def send(req):
            await ws.send(json.dumps(req))
            return json.loads(await ws.recv())

        await send({"id": 1, "jsonrpc": "2.0", "method": "requestAccess",
                    "params": {"clientId": "VxZIy6WsTOTxeS8ao9BIE1kG1eVKexHiId1HqMhr",
                               "clientSecret": "2DmGjonJv5FxU18cwFvkHFGV17605tu6i5HYu3VyDYm8E7m8ZpFyXDf81KBIl3RMcRj48S0tKi1OMAZYw792gqetsW5broSGjx1VTFcsoAIUWq8BU8LJjqlCousFyBUq"}})

        auth = await send({"id": 2, "jsonrpc": "2.0", "method": "authorize",
                           "params": {"clientId": "VxZIy6WsTOTxeS8ao9BIE1kG1eVKexHiId1HqMhr",
                                      "clientSecret": "2DmGjonJv5FxU18cwFvkHFGV17605tu6i5HYu3VyDYm8E7m8ZpFyXDf81KBIl3RMcRj48S0tKi1OMAZYw792gqetsW5broSGjx1VTFcsoAIUWq8BU8LJjqlCousFyBUq",
                                      "debit": 1}})

        if 'result' not in auth:
            print("❌ Authorization failed. Full response:", auth)
            return

        token = auth['result']['cortexToken']
        print(f"Token: {token}")

        session_response = await send({"id": 3, "jsonrpc": "2.0", "method": "createSession",
                                       "params": {"cortexToken": token, "headset": "EPOCX-E5020242",
                                                  "status": "active"}})

        if "result" not in session_response:
            print("❌ Failed to create session!")
            return

        session_id = session_response["result"]["id"]

        subscribe_response = await send({"id": 4, "jsonrpc": "2.0", "method": "subscribe",
                                         "params": {"cortexToken": token, "session": session_id, "streams": ["eeg"]}})

        while True:
            msg = await ws.recv()

            data = json.loads(msg)
            if "eeg" in data:
                #print("received")
                eeg = data.get("eeg", [])
                eeg = eeg[2:16]
                BUFFER.append(eeg)

                if not baseline_active:
                    process_baseline(BUFFER)

                if len(BUFFER) == WINDOW_SIZE:
                    print("start prediction")
                    predict_and_move(BUFFER)
                    BUFFER = []

# Setup
side_dots = generate_new_side_dots(full_sides)
current_action, remaining_side = get_remaining_side_and_action(side_dots)
new_action = ''
start_time_new_action = 0
start = time.time()

# Launch EEG thread
import threading

threading.Thread(target=lambda: asyncio.run(get_eeg_input()), daemon=True).start()

# Pygame Loop

running = True

# asyncio.run(get_eeg_input())

while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if baseline_active:
        elapsed = time.time() - baseline_time
        countdown = max(0, 30 - int(elapsed))
        text = font.render(f"Recording baseline: {countdown}s", True, WHITE)
        screen.blit(text, (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2))
        pygame.display.flip()
        if elapsed >= 30:
            baseline_active = False
            game_started = True
            start_time = time.time()
            print("Baseline done. Game started!")
            actions.append([baseline_time, time.time(), "baseline"])
        continue

    for dot in side_dots:
        pygame.draw.circle(screen, WHITE, dot, DOT_RADIUS)
    pygame.draw.circle(screen, RED, center_dot, DOT_RADIUS)

    if game_started:
        if reset:
            reset = False
            screen.fill(BLACK)
            #print('CURRENT ACTION:', current_action)
            #print('REMAINING SIDE DOT:', remaining_side)

            for dot in side_dots:
                pygame.draw.circle(screen, WHITE, dot, DOT_RADIUS)
            pygame.draw.circle(screen, RED, remaining_side, DOT_RADIUS)
            pygame.display.flip()
            # pygame.time.wait(500)
            pygame.draw.circle(screen, RED, center_dot, DOT_RADIUS)

            end_time_new_action = time.time()
            if start_time_new_action:
                actions.append([start_time_new_action, end_time_new_action, current_action])
            side_dots = generate_new_side_dots(full_sides)
            current_action, remaining_side = get_remaining_side_and_action(side_dots)
            start_time_new_action = time.time()
            screen.fill(BLACK)
            pygame.display.flip()
            start = time.time()

    pygame.display.flip()

# Save to CSV
filename = f'game_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['start_time_new_action', 'end_time_new_action', 'action'])
    writer.writerows(actions)

print(f"Data saved to {filename}")
pygame.quit()