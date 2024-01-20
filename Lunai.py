from mss import mss
import pyautogui
import cv2
import numpy as np
from tkinter import filedialog, Canvas, Entry, Button, PhotoImage, messagebox
import gymnasium as gym
from gymnasium import spaces
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
import torch
from stable_baselines3.common.env_checker import check_env
import subprocess
import webbrowser
import sys
import tkinter as tk
import time
import json
from tkinter import Toplevel
from PIL import Image, ImageDraw, ImageTk
import easyocr

# Check if CUDA (GPU support) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_hyperparameters():
    global bufferSize, learningStart, learningRate, batchSize, totalTimeStep, freqSave
    try:
        with open('advanced_settings.json', 'r') as f:
            hyperparams = json.load(f)
            bufferSize = hyperparams.get("bufferSize", 100000)
            learningStart = hyperparams.get("learningStart", 100)
            learningRate = hyperparams.get("learningRate", 0.00025)
            batchSize = hyperparams.get("batchSize", 32)
            totalTimeStep = hyperparams.get("totalTimeStep", 100000)
            freqSave = hyperparams.get("freqSave", 1000)
    except FileNotFoundError:
        bufferSize = 100000
        learningStart = 100
        learningRate = 0.00025
        batchSize = 32
        totalTimeStep = 100000
        freqSave = 1000


load_hyperparameters()


class Vision(gym.Env):
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.sct = mss()
        self.reader = easyocr.Reader(['en'])
        self.action_space = spaces.Discrete(2)
        self.doneString = [""]
        self.custom_reset_enabled = False
        self.custom_reset_coords = None
        self.reset_action = ('key', 'space')
        self.ai_view_enable = False
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.stop_training = False
        self.use_counter = False
        self.counter_reward = 0
        # Setup spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 74, 110), dtype=np.uint8)
        if os.path.exists('env_settings.json'):
            with open('env_settings.json', 'r') as f:
                settings = json.load(f)
                self.game_location = settings['game_location']
                self.done_location = settings['done_location']
                self.reward_location = settings['reward_location']
                self.doneString = settings['doneString']
                self.done_resize_dimensions = (self.done_location['width'], self.done_location['height'])
                self.reward_resize_dimensions = (self.reward_location['width'], self.reward_location['height'])
        self.action_map = {
            0: 'up',
            1: 'no_op'
        }

    def update_action_space(self, num_of_actions):
        self.action_space = spaces.Discrete(len(self.action_map))

    def update_observation_space(self, shape):
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def step(self, action):
        if action in self.action_map:
            event, key_to_press = self.action_map[action]
            if event == 'down':
                pyautogui.keyDown(key_to_press)
            elif event == 'up':
                pyautogui.keyUp(key_to_press)
        else:
            print(f"Action {action} not found in action map.")

        observation = self.get_observation(action)
        reward = self.get_reward()
        done = self.get_done()
        truncated = done
        info = {}
        return observation, reward, done, truncated, info

    def enable_custom_reset(self, enabled, coords=None):
        if enabled and coords:
            self.reset_action = ('click', coords)  # coords is a tuple (x, y)
        else:
            self.reset_action = ('key', 'space')

    def reset(self, seed=None, **kwargs):
        action_type, action_value = self.reset_action
        if action_type == 'click':
            x, y = action_value
            pyautogui.click(x, y)
        elif action_type == 'key':
            pyautogui.press(action_value)
        print(f"Reset action initiated: {self.reset_action}")
        self.counter_reward = 0
        info = {}
        return self.get_observation(), info

    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    def capture_image(self, location):
        return np.array(self.sct.grab(location))[:, :, :3].astype(np.uint8)

    def resize_image(self, image):
        _, height, width = self.observation_space.shape
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.reshape(resized, (1, height, width))

    def process_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def update_fps(self):
        current_time = time.time()
        self.frame_count += 1
        elapsed_time = current_time - self.last_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_time = current_time

    def display_ai_view(self, image, action):
        font_size = 0.4  # Smaller font size
        thickness = 1  # Reduced thickness
        cv2.putText(image, f"FPS: {self.fps:.2f}", (1, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), thickness)
        action_word = self.action_map.get(action, "Unknown") if action is not None else "None"
        cv2.putText(image, f"Action: {action_word}", (1, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), thickness)

        window_name = "AI View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    def get_observation(self, action=None):
        raw = self.capture_image(self.game_location)
        gray = self.process_image(raw)
        channel = self.resize_image(gray)
        display_image = channel[0, :, :]

        if self.ai_view_enable:
            self.update_fps()
            self.display_ai_view(display_image, action)
        return channel

    def get_done(self):
        done_str = self.get_balance()
        if done_str in self.doneString:
            print("Resetting")
            return True
        if done_str == self.doneString:
            print("Resetting")
            return True
        return False

    def get_balance(self):
        raw = self.capture_image(self.done_location)
        try:
            result = self.reader.readtext(raw)
            done_str = ''.join([text for _, text, _ in result])[:len(self.doneString)]
            print(done_str)
            return done_str
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""

    def get_reward(self):
        if not self.use_counter:
            raw = self.capture_image(self.reward_location)
            try:
                result = self.reader.readtext(raw)
                balance_str = ''.join([text for _, text, _ in result])
                balance = float(balance_str)
                print("Reward = " + str(balance))
                return balance
            except Exception as e:
                return 0
        else:
            self.counter_reward += 1
            print("Reward = " + str(self.counter_reward))
            return self.counter_reward


def show_preview(game_location, done_location, reward_location):
    # Create an overlay image
    screenshot = pyautogui.screenshot()
    overlay_image = Image.new("RGBA", screenshot.size)
    overlay_image.paste(screenshot, (0, 0))
    draw = ImageDraw.Draw(overlay_image)

    draw.rectangle([(game_location['left'], game_location['top']),
                    (game_location['left'] + game_location['width'], game_location['top'] + game_location['height'])],
                   outline="blue", width=5)
    draw.rectangle([(done_location['left'], done_location['top']),
                    (done_location['left'] + done_location['width'], done_location['top'] + done_location['height'])],
                   outline="red", width=5)
    draw.rectangle([(reward_location['left'], reward_location['top']),
                    (reward_location['left'] + reward_location['width'],
                     reward_location['top'] + reward_location['height'])],
                   outline="green", width=5)

    overlay_image.show()


def update_parameters(env, new_game_location, new_done_location, new_reward_location, new_doneString):
    env.game_location = json.loads(new_game_location)
    env.done_location = json.loads(new_done_location)
    env.reward_location = json.loads(new_reward_location)
    env.doneString = json.loads(new_doneString)


# Color Scheme
bg_color = "#2e2e2e"  # Dark gray background
text_color = "#ffffff"  # White text
button_color = "#4a4a4a"  # Lighter gray for buttons
entry_bg = "#3a3a3a"  # Slightly lighter gray for entry fields
entry_fg = "#ffffff"  # White text for entries

# Common Styles
font_style = "Helvetica"
font_size = 12


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def exit_application():
    sys.exit()


class AccountSetting:
    def __init__(self, env):
        self.new_model = None
        self.env = env
        self.root = tk.Tk()
        self.root.title("Lunai")
        self.root.geometry("636x422")
        self.root.configure(bg="#FFFFFF")
        self.loadedModel = False
        self.playModel = False
        self.modelPath = ""

        # Start with the login screen
        self.create_login_gui()
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", exit_application)

    def create_main_gui(self):
        canvas = Canvas(
            self.root,
            bg="#FFFFFF",
            height=432,
            width=636,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=-10)
        image_image_1 = PhotoImage(
            file=resource_path("assets/frame0/image_1.png"))
        image_1 = canvas.create_image(
            318.0,
            225.0,
            image=image_image_1
        )

        button_image_1 = PhotoImage(
            file=resource_path("assets/frame0/button_1.png"))
        button_1 = Button(
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.load_model,
            relief="flat"
        )
        button_1.place(
            x=53.0,
            y=172.0,
            width=156.0,
            height=41.0
        )

        button_image_2 = PhotoImage(
            file=resource_path("assets/frame0/button_2.png"))
        button_2 = Button(
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.play_model,
            relief="flat"
        )
        button_2.place(
            x=53.0,
            y=106.0,
            width=156.0,
            height=41.0
        )

        button_image_3 = PhotoImage(
            file=resource_path("assets/frame0/button_3.png"))
        button_3 = Button(
            image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=self.go_to_action_settings,
            relief="flat"
        )
        button_3.place(
            x=53.0,
            y=350.0,
            width=156.0,
            height=41.0
        )

        button_image_4 = PhotoImage(
            file=resource_path("assets/frame0/button_4.png"))
        button_4 = Button(
            self.root,
            image=button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=self.toggle_win,
            relief="flat"
        )
        button_4.place(
            x=549.0,
            y=28.0,
            width=50.0,
            height=50.0
        )

        canvas.create_text(
            53.0,
            53.0,
            anchor="nw",
            text="Choose an action : ",
            fill="#FFFFFF",
            font=("Inter", 32 * -1)
        )

        canvas.create_text(
            53.0,
            23.0,
            anchor="nw",
            text="Hello User,",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        self.root.mainloop()

    def create_login_gui(self):
        self.canvas = Canvas(
            self.root,  # Use the main window instead of creating a new Tk instance
            bg="#FFFFFF",
            height=432,
            width=636,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas.place(x=0, y=0)

        self.image_image_1 = PhotoImage(
            file=resource_path("assets/frame0/image_1.4.png"))
        image_1 = self.canvas.create_image(
            186.0,
            216.0,
            image=self.image_image_1
        )

        self.button_image_1 = PhotoImage(
            file=resource_path("assets/frame0/button_1.4.png"))
        button_1 = Button(
            self.root,
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.create_main_gui,
            relief="flat"
        )
        button_1.place(
            x=95.0,
            y=140.0,
            width=201.0,
            height=62.0
        )

        self.button_license = PhotoImage(
            file=resource_path("assets/frame0/license.png"))
        button_license = Button(
            self.root,
            image=self.button_license,
            borderwidth=0,
            highlightthickness=0,
            command=self.donation,
            relief="flat"
        )
        button_license.place(
            x=130.0,
            y=238.0,
            width=127.0,
            height=44.0
        )

        self.canvas.create_text(
            53.0,
            53.0,
            anchor="nw",
            text="Press begin,",
            fill="#FFFFFF",
            font=("Inter", 32 * -1)
        )

        self.canvas.create_text(
            25.0,
            20.0,
            anchor="nw",
            text="Lunai",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        self.image_image_2 = PhotoImage(
            file=resource_path("assets/frame0/image_2.4.png"))
        image_2 = self.canvas.create_image(
            507.0,
            307.0,
            image=self.image_image_2
        )

        # Load and resize the GIF
        self.original_gif = Image.open(resource_path("assets/frame0/image_3.44.gif"))
        self.resized_gif = self.original_gif.resize((175, 106), Image.Resampling.LANCZOS)
        self.gif_frame_index = 0

        self.gif_photoimage = ImageTk.PhotoImage(self.resized_gif)
        self.gif_label = tk.Label(self.root, image=self.gif_photoimage)
        self.gif_label.place(x=425, y=221)

        # Start the GIF animation
        self.animate_gif()

        self.image_image_4 = PhotoImage(
            file=resource_path("assets/frame0/image_4.4.png"))
        image_4 = self.canvas.create_image(
            507.0,
            91.0,
            image=self.image_image_4
        )

        self.canvas.create_rectangle(
            373.0,
            0.0,
            378.0,
            441.0,
            fill="#D9D9D9",
            outline="")

        # Add a label for error messages
        self.error_label = tk.Label(self.root, text="", fg="red", bg="#1D1129")
        self.error_label.place(x=95, y=250)

    def while_running_ai(self):
        print("Worked")

        canvas = Canvas(
            self.root,
            bg="#FFFFFF",
            height=432,
            width=636,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=-10)
        image_image_1 = PhotoImage(
            file=resource_path("assets/frame0/image_1.png"))
        image_1 = canvas.create_image(
            318.0,
            225.0,
            image=image_image_1
        )

    def animate_gif(self):
        try:
            self.original_gif.seek(self.gif_frame_index)
            self.resized_gif = self.original_gif.resize((175, 105), Image.Resampling.LANCZOS)
            self.gif_photoimage = ImageTk.PhotoImage(self.resized_gif)
            self.gif_label.configure(image=self.gif_photoimage)
            self.gif_frame_index += 1
        except EOFError:
            # Reset to the first frame
            self.gif_frame_index = 0
            self.original_gif.seek(self.gif_frame_index)
        finally:
            # Continue the loop after a delay (e.g., 100 milliseconds)
            self.root.after(50, self.animate_gif)

    def clear_placeholder(self, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg='black')

    def add_placeholder(self, entry, placeholder):
        if not entry.get():
            entry.insert(0, placeholder)
            entry.config(fg='grey')

    def donation(self):
        webbrowser.open("https://linktr.ee/AiLunai", new=2)

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.zip")])
        if file_path:
            self.modelPath = file_path
            print(f"Model loaded from {file_path}")
            self.loadedModel = True
            self.go_to_action_settings()

    def play_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.zip")])
        if file_path:
            self.modelPath = file_path
            print(f"Model loaded from {file_path}")
            self.playModel = True
            self.go_to_action_settings()

    def toggle_win(self):
        # Set initial position of the frame outside the right edge of the root window
        initial_x = self.root.winfo_width()
        self.f1 = tk.Frame(self.root, width=300, height=500, bg='#262626')
        self.f1.place(x=initial_x, y=0)

        # Load the image and apply it as a background
        self.bg_image = PhotoImage(file='assets/frame0/image_1.3.png')
        bg_label = tk.Label(self.f1, image=self.bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        def bttn(x, y, text, bcolor, fcolor, cmd):

            def on_enters(e):
                e.widget['background'] = bcolor  # Change the color of the widget that triggered the event
                e.widget['foreground'] = '#FFFFFF'

            def on_leaves(e):
                e.widget['background'] = fcolor  # Revert the color of the widget that triggered the event
                e.widget['foreground'] = '#FFFFFF'

            myButton = Button(self.f1, text=text,
                              width=42,
                              height=2,
                              fg='#FFFFFF',
                              border=0,
                              bg=fcolor,
                              activebackground='#262626',
                              activeforeground=bcolor,
                              command=cmd)
            myButton.bind("<Enter>", on_enters)
            myButton.bind("<Leave>", on_leaves)

            myButton.place(x=x, y=y)

        bttn(0, 100, 'L O A D   L O G S', '#39A9E0', '#42095D', self.open_tensorboard)
        bttn(0, 340, 'L O G   O U T', '#39A9E0', '#42095D', self.create_login_gui)

        def slide_in():
            # Slide the frame in from the right
            x = initial_x
            while x > self.root.winfo_width() - 300:  # Adjust 300 to the width of the frame
                x -= 10  # Adjust the step size for smoother or faster animation
                self.f1.place(x=x, y=0)
                self.f1.update()

        def slide_out():
            # Slide the frame out to the right and destroy
            x = self.root.winfo_width() - 300  # Adjust 300 to the width of the frame
            while x < initial_x:
                x += 10  # Adjust the step size for smoother or faster animation
                self.f1.place(x=x, y=0)
                self.f1.update()
            self.f1.destroy()

        def on_root_click(event):
            # Check if the f1 frame exists and is not destroyed
            if self.f1.winfo_exists():
                x1, y1, x2, y2 = self.f1.winfo_rootx(), self.f1.winfo_rooty(), self.f1.winfo_rootx() + self.f1.winfo_width(), self.f1.winfo_rooty() + self.f1.winfo_height()
                if not (x1 < event.x_root < x2 and y1 < event.y_root < y2):
                    slide_out()

        slide_in()  # Slide in the frame

        # Bind click event to the root window
        self.root.bind("<Button-1>", on_root_click)

    def do_nothing(self):
        pass

    def open_tensorboard(self):
        # Ask the user to select the Tensorboard log directory
        log_dir = filedialog.askdirectory(title="Select Tensorboard Log Directory")
        if log_dir:
            # Start Tensorboard pointing to the selected directory
            tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir])
            # Open the browser to view Tensorboard
            webbrowser.open("http://localhost:6006", new=2)

    def go_to_action_settings(self):
        ActionSettingGUI(self.env)

    def close_application(self):
        self.root.destroy()


class ActionSettingGUI:
    def __init__(self, env):
        self.env = env
        self.root = tk.Toplevel()
        self.root.title("AI Configuration")
        self.root.geometry("636x243")
        self.root.configure(bg="#FFFFFF")

        self.current_step = "num_actions"  # Start with setting number of actions
        self.num_of_actions = 0
        self.current_action = 0
        self.action_map_entries = {}

        canvas = Canvas(
            self.root,
            bg="#FFFFFF",
            height=243,
            width=636,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        self.canvas = canvas

        canvas.place(x=0, y=0)
        image_image_1 = PhotoImage(
            file=resource_path("assets/frame0/image_1.1.png"))
        image_1 = canvas.create_image(
            318.0,
            121.0,
            image=image_image_1
        )

        button_image_1 = PhotoImage(
            file=resource_path("assets/frame0/button_1.1.png"))
        button_1 = Button(
            self.root,
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.go_back,
            relief="flat"
        )
        button_1.place(
            x=60.0,
            y=156.0,
            width=156.0,
            height=41.0
        )

        button_image_2 = PhotoImage(
            file=resource_path("assets/frame0/button_2.1.png"))
        button_2 = Button(
            self.root,
            image=button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.go_next,
            relief="flat"
        )
        button_2.place(
            x=414.0,
            y=156.0,
            width=156.0,
            height=41.0
        )

        canvas.create_text(
            53.0,
            23.0,
            anchor="nw",
            text="Hello User,",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        self.instruction_label = canvas.create_text(
            53.0,
            53.0,
            anchor="nw",
            text="",
            fill="#FFFFFF",
            font=("Inter", 32 * -1)
        )

        self.instruction_label2 = canvas.create_text(
            53.0,
            89.0,
            anchor="nw",
            text="",
            fill="#FFFFFF",
            font=("Inter", 32 * -1)
        )

        self.pressed_key_label = canvas.create_text(
            53.0,
            89.0,
            anchor="nw",
            text="",
            fill="#FFFFFF",
            font=("Inter", 32 * -1)
        )

        self.num_actions_entry = tk.Entry(self.root, font=(font_style, font_size + 12), bg=entry_bg, fg=entry_fg)
        self.num_actions_entry.place(x=260.0, y=90.0, width=81.0, height=38.0)

        self.root.bind("<Key>", self.key_pressed)
        self.update_gui()
        self.root.resizable(False, False)
        self.root.mainloop()

    def update_gui(self):
        if self.current_step == "num_actions":
            self.canvas.itemconfig(self.instruction_label, text="Enter number of actions you want")
            self.canvas.itemconfig(self.instruction_label2, text="the AI to take :")
            self.canvas.itemconfig(self.pressed_key_label, text="")
            self.num_actions_entry.place()
        else:
            self.canvas.itemconfig(self.instruction_label, text=f"Press a key for action {self.current_action + 1}")
            self.canvas.itemconfig(self.instruction_label2, text="")
            self.num_actions_entry.place_forget()
            entry = self.action_map_entries.get(self.current_action)
            if entry:
                self.canvas.itemconfig(self.pressed_key_label, text=f"Pressed: {entry.get()}")
            else:
                self.canvas.itemconfig(self.pressed_key_label, text="")

    def key_pressed(self, event):
        if self.current_step == "set_actions":
            self.action_map_entries[self.current_action] = tk.StringVar(value=event.keysym)
            self.canvas.itemconfig(self.pressed_key_label, text=f"Pressed: {event.keysym}")

    def go_next(self):
        if self.current_step == "num_actions":
            try:
                self.num_of_actions = int(self.num_actions_entry.get())
                self.current_step = "set_actions"
                self.update_gui()
            except ValueError:
                print("Please enter a valid integer for the number of actions.")
        elif self.current_action < self.num_of_actions - 1:
            # If no key has been pressed for the current action, assign "no-op"
            if self.current_action not in self.action_map_entries:
                self.action_map_entries[self.current_action] = tk.StringVar(value="no-op")
            self.current_action += 1
            self.update_gui()
        else:
            # Check for the last action
            if self.current_action not in self.action_map_entries:
                self.action_map_entries[self.current_action] = tk.StringVar(value="no-op")
            self.finish_setting()

    def go_back(self):
        if self.current_action > 0:
            self.current_action -= 1
            self.update_gui()

    def finish_setting(self):
        new_action_map = {action: entry.get() for action, entry in self.action_map_entries.items()}
        for action_index, key_name in new_action_map.items():
            if key_name:
                self.env.action_map[2 * action_index] = ('down', key_name)  # Action for pressing the key down
                self.env.action_map[2 * action_index + 1] = ('up', key_name)  # Action for releasing the key
        self.env.update_action_space(len(new_action_map))
        print("New action map set:", self.env.action_map)
        self.root.destroy()
        create_main_gui(self.env)


def create_main_gui(env):
    root = tk.Toplevel()
    root.title("Environment Parameters")
    root.geometry("636x432")
    root.configure(bg="#FFFFFF")
    keyy = ""
    keyyy = None

    large_font = (font_style, font_size + 2)
    small_font = (font_style, font_size - 2)

    canvas = Canvas(
        root,
        bg="#FFFFFF",
        height=432,
        width=636,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas.place(x=0, y=0)
    image_image_1 = PhotoImage(
        file=resource_path("assets/frame0/image_1.2.png"))
    image_1 = canvas.create_image(
        318.0,
        225.0,
        image=image_image_1
    )

    def submit():
        new_game_location = game_location_entry.get()
        new_done_location = done_location_entry.get()
        new_reward_location = reward_location_entry.get()
        new_done_string = doneString_entry.get()
        update_parameters(env, new_game_location, new_done_location, new_reward_location, new_done_string)
        with open('env_settings.json', 'w') as f:
            json.dump({
                'game_location': env.game_location,
                'done_location': env.done_location,
                'reward_location': env.reward_location,
                'doneString': env.doneString
            }, f, indent=4)
        root.destroy()
        ai = StartAi(env, account_setting)
        ai.run()

    def preview():
        new_game_location = json.loads(game_location_entry.get())
        new_done_location = json.loads(done_location_entry.get())
        new_reward_location = json.loads(reward_location_entry.get())
        show_preview(new_game_location, new_done_location, new_reward_location)

    def go_back():
        root.destroy()
        ActionSettingGUI(env)

    def advanced_settings():
        advanced_root = tk.Toplevel()
        advanced_root.title("Advanced Settings")
        advanced_root.geometry("348x670")
        advanced_root.configure(bg="#FFFFFF")

        canvas = Canvas(
            advanced_root,
            bg="#FFFFFF",
            height=670,
            width=348,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )

        canvas.place(x=0, y=0)
        image_image_1 = PhotoImage(
            file=resource_path("assets/frame0/image_1.3.png"))
        image_1 = canvas.create_image(
            174.0,
            344.0,
            image=image_image_1
        )

        # Variable to store coordinates
        coords_var = tk.StringVar(value="")

        # Checkbox variable
        custom_reset_var = tk.BooleanVar(value=env.custom_reset_enabled)
        ai_view_var = tk.BooleanVar(value=env.ai_view_enable)
        counter_checkbox_var = tk.BooleanVar(value=env.use_counter)

        # Function to capture click coordinates
        def capture_click(event):
            x, y = event.x_root, event.y_root
            coords_var.set(f"{x}, {y}")
            env.enable_custom_reset(True, (x, y))
            advanced_root.overlay.destroy()
            advanced_root.focus_force()

        # Function to create overlay
        def create_overlay():
            advanced_root.overlay = Toplevel(advanced_root)
            advanced_root.overlay.attributes('-fullscreen', True)
            advanced_root.overlay.attributes('-alpha', 0.3)
            advanced_root.overlay.configure(bg='grey')
            advanced_root.overlay.bind('<Button-1>', capture_click)

        # Checkbox toggle function
        def on_checkbox_toggle():
            is_checked = custom_reset_var.get()
            env.enable_custom_reset(is_checked)
            if is_checked:
                create_overlay()
            else:
                coords_var.set("")

        def ai_view_toggle():
            is_checked = ai_view_var.get()
            env.ai_view_enable = is_checked

        def counter_reward():
            is_checked = counter_checkbox_var.get()
            env.use_counter = is_checked

        # Function to save hyperparameters
        def save_advanced_settings():
            global bufferSize, learningStart, learningRate, batchSize, totalTimeStep, freqSave
            try:
                bufferSize = int(bufferSize_entry.get())
                learningStart = int(learningStart_entry.get())
                learningRate = float(learningRate_entry.get())
                batchSize = int(batchSize_entry.get())
                totalTimeStep = int(totalTimeStep_entry.get())
                freqSave = int(freqSave_entry.get())

                hyperparams = {
                    "bufferSize": bufferSize,
                    "learningStart": learningStart,
                    "learningRate": learningRate,
                    "batchSize": batchSize,
                    "totalTimeStep": totalTimeStep,
                    "freqSave": freqSave,
                }

                with open('advanced_settings.json', 'w') as f:
                    json.dump(hyperparams, f, indent=4)

            except ValueError as e:
                tk.messagebox.showerror("Invalid Input", "Please enter valid hyperparameter values.")

            # Save observation shape
            try:
                obs_shape = json.loads(obs_shape_entry.get())
                if len(obs_shape) == 3 and all(isinstance(dim, int) for dim in obs_shape):
                    env.update_observation_space(tuple(obs_shape))
                    advanced_root.destroy()
                else:
                    raise ValueError("Invalid observation shape")
            except ValueError as e:
                tk.messagebox.showerror("Invalid Input", "Please enter valid values for observation shape.")

        # Creating entry fields for hyperparameters
        canvas.create_text(
            86.0,
            51.0,
            anchor="nw",
            text="size ",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            72.0,
            32.0,
            anchor="nw",
            text="Buffer",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_1 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_1 = canvas.create_image(
            213.5,
            51.0,
            image=entry_image_1
        )
        bufferSize_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        bufferSize_entry.place(
            x=118.0,
            y=39.0,
            width=191.0,
            height=22.0
        )
        bufferSize_entry.insert(0, json.dumps(bufferSize))

        canvas.create_text(
            53.0,
            78.0,
            anchor="nw",
            text="Learning  ",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            81.0,
            96.0,
            anchor="nw",
            text="start ",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_2 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_2 = canvas.create_image(
            213.5,
            96.0,
            image=entry_image_2
        )
        learningStart_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        learningStart_entry.place(
            x=118.0,
            y=84.0,
            width=191.0,
            height=22.0
        )
        learningStart_entry.insert(0, json.dumps(learningStart))

        canvas.create_text(
            53.0,
            124.0,
            anchor="nw",
            text="Learning  ",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            86.0,
            142.0,
            anchor="nw",
            text="rate ",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_3 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_3 = canvas.create_image(
            213.5,
            142.0,
            image=entry_image_3
        )
        learningRate_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        learningRate_entry.place(
            x=118.0,
            y=130.0,
            width=191.0,
            height=22.0
        )
        learningRate_entry.insert(0, json.dumps(learningRate))

        canvas.create_text(
            75.0,
            170.0,
            anchor="nw",
            text="Batch",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            86.0,
            188.0,
            anchor="nw",
            text="size",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_4 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_4 = canvas.create_image(
            213.5,
            188.0,
            image=entry_image_4
        )
        batchSize_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        batchSize_entry.place(
            x=118.0,
            y=176.0,
            width=191.0,
            height=22.0
        )
        batchSize_entry.insert(0, json.dumps(batchSize))

        canvas.create_text(
            81.0,
            216.0,
            anchor="nw",
            text="Total",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            46.0,
            234.0,
            anchor="nw",
            text="timesteps",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_5 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_5 = canvas.create_image(
            213.5,
            234.0,
            image=entry_image_5
        )
        totalTimeStep_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        totalTimeStep_entry.place(
            x=118.0,
            y=222.0,
            width=191.0,
            height=22.0
        )
        totalTimeStep_entry.insert(0, json.dumps(totalTimeStep))

        canvas.create_text(
            81.0,
            262.0,
            anchor="nw",
            text="Save",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            43.0,
            280.0,
            anchor="nw",
            text="frequency",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_7 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_7 = canvas.create_image(
            213.5,
            280.0,
            image=entry_image_7
        )
        freqSave_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        freqSave_entry.place(
            x=118.0,
            y=268.0,
            width=191.0,
            height=22.0
        )
        freqSave_entry.insert(0, json.dumps(freqSave))

        # Entries for observation shape
        canvas.create_text(
            29.0,
            308.0,
            anchor="nw",
            text="Observation",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )

        canvas.create_text(
            72.0,
            326.0,
            anchor="nw",
            text="shape",
            fill="#FFFFFF",
            font=("Inter", 15 * -1)
        )
        entry_image_6 = PhotoImage(
            file=resource_path("assets/frame0/entry_1.3.png"))
        entry_bg_6 = canvas.create_image(
            213.5,
            326.0,
            image=entry_image_6
        )
        obs_shape_entry = Entry(
            advanced_root,
            bd=0,
            bg="#42095D",
            fg=text_color,
            highlightthickness=0
        )
        obs_shape_entry.place(
            x=118.0,
            y=314.0,
            width=191.0,
            height=22.0
        )
        obs_shape_entry.insert(0, json.dumps(env.observation_space.shape))
        tk.Label(advanced_root,
                 text="(Setting your observation too high will slow down your Ai and use more RAM. Setting your observation too low will make your Ai blind)",
                 bg=bg_color, fg=text_color,
                 anchor='w', wraplength=250).pack(pady=(355, 0))

        # Initialize the coordinates label if custom reset is already enabled
        if env.custom_reset_coords:
            coords_var.set(f"{env.custom_reset_coords[0]}, {env.custom_reset_coords[1]}")

        custom_reset_checkbox = tk.Checkbutton(advanced_root, text="Enable reset click?", var=custom_reset_var,
                                               command=on_checkbox_toggle, font=(font_style, font_size), bg=bg_color,
                                               fg=text_color, selectcolor="black")
        custom_reset_checkbox.pack(pady=(10, 0))

        coords_label = tk.Label(advanced_root, textvariable=coords_var, font=(font_style, font_size), bg="#2C122B",
                                fg=text_color)
        coords_label.pack(pady=5)

        ai_view_checkbox = tk.Checkbutton(advanced_root, text="Enable Ai view?", var=ai_view_var,
                                          command=ai_view_toggle, font=(font_style, font_size), bg=bg_color,
                                          fg=text_color, selectcolor="black")
        ai_view_checkbox.pack(pady=(10, 0))

        counter_checkbox = tk.Checkbutton(advanced_root, text="Use a counter as reward instead",
                                          var=counter_checkbox_var,
                                          command=counter_reward, font=(font_style, font_size), bg=bg_color,
                                          fg=text_color, selectcolor="black")
        counter_checkbox.pack(pady=(10, 0))

        # Save button
        button_image_1 = PhotoImage(
            file=resource_path("assets/frame0/button_1.3.png"))
        button_1 = Button(
            advanced_root,
            image=button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=save_advanced_settings,
            relief="flat"
        )
        button_1.place(
            x=96.0,
            y=596.0,
            width=156.0,
            height=41.0
        )

        advanced_root.resizable(False, False)
        advanced_root.mainloop()

    def select_area(env, setting_name, entry_widget):
        start_x, start_y, end_x, end_y = 0, 0, 0, 0

        def on_click(event):
            nonlocal start_x, start_y
            start_x, start_y = event.x_root, event.y_root
            canvas.coords('drag_rectangle', start_x, start_y, start_x, start_y)

        def on_drag(event):
            nonlocal end_x, end_y
            end_x, end_y = event.x_root, event.y_root
            draw_rectangle(canvas, start_x, start_y, end_x, end_y)

        def on_release(event):
            nonlocal start_x, start_y, end_x, end_y
            update_setting(env, setting_name, start_x, start_y, end_x, end_y, entry_widget)
            selection_overlay.destroy()

        selection_overlay = tk.Toplevel()
        selection_overlay.attributes('-fullscreen', True, '-alpha', 0.3)
        # Create a transparent canvas for drawing the rectangle
        canvas = tk.Canvas(selection_overlay, cursor="cross")
        canvas.pack(fill=tk.BOTH, expand=True)
        # Create a rectangle on the canvas (initially invisible)
        canvas.create_rectangle(0, 0, 0, 0, outline='red', tag='drag_rectangle')
        # Bind the mouse events
        canvas.bind('<Button-1>', on_click)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_release)
        selection_overlay.mainloop()

    def draw_rectangle(canvas, x1, y1, x2, y2):
        canvas.coords('drag_rectangle', x1, y1, x2, y2)

    def update_setting(env, setting_name, x1, y1, x2, y2, entry_widget):
        coords = {'top': min(y1, y2), 'left': min(x1, x2), 'width': abs(x2 - x1), 'height': abs(y2 - y1)}
        setattr(env, setting_name, coords)
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, json.dumps(coords))
        if setting_name == 'game_location':
            env.reset_action = ('click', (min(x1, x2), min(y1, y2)))

    canvas.create_text(
        53.0,
        34.0,
        anchor="nw",
        text="\nlocation :",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )

    canvas.create_text(
        53.0,
        34.0,
        anchor="nw",
        text="Window",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )
    entry_image_1 = PhotoImage(
        file=resource_path("assets/frame0/entry_1.2.png"))
    entry_bg_1 = canvas.create_image(
        292.0,
        60.0,
        image=entry_image_1
    )
    game_location_entry = Entry(
        root,
        bd=0,
        bg="#42095D",
        fg=entry_fg,
        highlightthickness=0
    )
    game_location_entry.place(
        x=123.0,
        y=48.0,
        width=338.0,
        height=22.0
    )
    game_location_entry.insert(0, json.dumps(env.game_location))

    button_image_1 = PhotoImage(
        file=resource_path("assets/frame0/button_1.2.png"))
    button_1 = Button(
        root,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: select_area(env, "game_location", game_location_entry),
        relief="flat"
    )
    button_1.place(
        x=476.0,
        y=45.0,
        width=130.0,
        height=30.0
    )

    canvas.create_text(
        53.0,
        97.0,
        anchor="nw",
        text="Finish  ",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )

    canvas.create_text(
        53.0,
        115.0,
        anchor="nw",
        text="location :",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )
    entry_image_2 = PhotoImage(
        file=resource_path("assets/frame0/entry_2.2.png"))
    entry_bg_2 = canvas.create_image(
        292.0,
        124.0,
        image=entry_image_2
    )
    done_location_entry = Entry(
        root,
        bd=0,
        bg="#42095D",
        fg=entry_fg,
        highlightthickness=0
    )
    done_location_entry.place(
        x=123.0,
        y=112.0,
        width=338.0,
        height=22.0
    )
    done_location_entry.insert(0, json.dumps(env.done_location))

    button_image_2 = PhotoImage(
        file=resource_path("assets/frame0/button_2.2.png"))
    button_2 = Button(
        root,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: select_area(env, "finish_location", done_location_entry),
        relief="flat"
    )
    button_2.place(
        x=476.0,
        y=109.0,
        width=130.0,
        height=30.0
    )

    canvas.create_text(
        54.0,
        178.0,
        anchor="nw",
        text="location :",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )

    canvas.create_text(
        54.0,
        160.0,
        anchor="nw",
        text="Score ",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )
    entry_image_3 = PhotoImage(
        file=resource_path("assets/frame0/entry_3.2.png"))
    entry_bg_3 = canvas.create_image(
        292.0,
        187.0,
        image=entry_image_3
    )
    reward_location_entry = Entry(
        root,
        bd=0,
        bg="#42095D",
        fg=entry_fg,
        highlightthickness=0
    )
    reward_location_entry.place(
        x=123.0,
        y=175.0,
        width=338.0,
        height=22.0
    )
    reward_location_entry.insert(0, json.dumps(env.reward_location))

    button_image_3 = PhotoImage(
        file=resource_path("assets/frame0/button_3.2.png"))
    button_3 = Button(
        root,
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: select_area(env, "done_location", reward_location_entry),
        relief="flat"
    )
    button_3.place(
        x=476.0,
        y=172.0,
        width=130.0,
        height=30.0
    )

    canvas.create_text(
        55.0,
        223.0,
        anchor="nw",
        text="Finish location identifier :",
        fill="#FFFFFF",
        font=("Inter", 15 * -1)
    )
    entry_image_4 = PhotoImage(
        file=resource_path("assets/frame0/entry_4.2.png"))
    entry_bg_4 = canvas.create_image(
        142.5,
        265.0,
        image=entry_image_4
    )
    doneString_entry = Entry(
        root,
        bd=0,
        bg="#42095D",
        fg=entry_fg,
        highlightthickness=0
    )
    doneString_entry.place(
        x=53.0,
        y=253.0,
        width=179.0,
        height=22.0
    )
    doneString_entry.insert(0, json.dumps(env.doneString))

    button_image_4 = PhotoImage(
        file=resource_path("assets/frame0/button_4.2.png"))
    button_4 = Button(
        root,
        image=button_image_4,
        borderwidth=0,
        highlightthickness=0,
        command=preview,
        relief="flat"
    )
    button_4.place(
        x=461.0,
        y=244.0,
        width=156.0,
        height=41.0
    )

    button_image_6 = PhotoImage(
        file=resource_path("assets/frame0/button_6.2.png"))
    button_6 = Button(
        root,
        image=button_image_6,
        borderwidth=0,
        highlightthickness=0,
        command=go_back,
        relief="flat"
    )
    button_6.place(
        x=40.0,
        y=353.0,
        width=156.0,
        height=41.0
    )

    button_image_5 = PhotoImage(
        file=resource_path("assets/frame0/button_5.2.png"))
    button_5 = Button(
        root,
        image=button_image_5,
        borderwidth=0,
        highlightthickness=0,
        command=submit,
        relief="flat"
    )
    button_5.place(
        x=240.0,
        y=312.0,
        width=156.0,
        height=41.0
    )

    button_image_7 = PhotoImage(
        file=resource_path("assets/frame0/button_7.2.png"))
    button_7 = Button(
        root,
        image=button_image_7,
        borderwidth=0,
        highlightthickness=0,
        command=advanced_settings,
        relief="flat"
    )
    button_7.place(
        x=450.0,
        y=353.0,
        width=156.0,
        height=41.0
    )

    root.resizable(False, False)
    root.mainloop()


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join('models', 'best_model_{}'.format(self.n_calls))
            print(f"Saving model at {model_path}")
            self.model.save(model_path)
            print('Model Saved')
        return True


class StartAi:
    def __init__(self, env, account_setting):
        self.env = env
        self.account_setting = account_setting
        self.model = None
        self.stop_training = False

    def run(self):
        LOG_DIR = './logs/'
        CHECKPOINT_DIR = './models/'
        callback = TrainAndLoggingCallback(check_freq=freqSave, save_path=CHECKPOINT_DIR)
        check_env(env)
        batch_size = batchSize
        learning_rate = learningRate

        if self.account_setting.loadedModel:
            model_path = self.account_setting.modelPath
            self.model = DQN.load(model_path)
            self.model.set_env(self.env)
            self.model.exploration_schedule = lambda _: 0.05
            print("Model loaded from " + model_path)
            self.model.learn(total_timesteps=totalTimeStep, callback=callback)

        elif self.account_setting.playModel:
            model_path = self.account_setting.modelPath
            self.model = DQN.load(model_path)
            self.model.set_env(self.env)
            print("Playing model")
            for episode in range(10):
                obs, info = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, truncated, info = self.env.step(int(action))
                    total_reward += reward
                print('Total Reward for episode {} is {}'.format(episode, total_reward))
        else:
            self.model = DQN('CnnPolicy', self.env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=bufferSize,
                             learning_starts=learningStart)
            print("New model loaded")
            while not self.stop_training:
                self.model.learn(total_timesteps=totalTimeStep, callback=callback)
                if self.stop_training:
                    break

        self.account_setting.create_main_gui()


env = Vision()
account_setting = AccountSetting(env)
account_setting.root.mainloop()
