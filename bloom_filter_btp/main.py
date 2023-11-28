import json
import os
import tkinter as tk
from functools import partial
from tkinter import messagebox

import cv2 as cv
from icecream import ic
from PIL import Image, ImageTk

from bloom_filter_btp.btp import (
    MIN_NEIGHBORS,
    SCALE_FACTOR,
    TEMPLATE_PATH,
    compare_bloom_filters,
    gen_template,
    save_user_template,
)
from bloom_filter_btp.img_processing import capture_and_extract_face

REFRESH_TIME = 1


def register_user(
    entry: tk.Entry,
    users_list: tk.Listbox,
) -> None:
    global templates, face

    if face is None:
        messagebox.showwarning("Warning", "No face detected")
        return

    username = entry.get()
    if username:
        print(f"Registering user: {username}")
        users_list.insert(tk.END, username)  # Add the username to the listbox
        entry.delete(0, tk.END)

        template = gen_template(face)
        save_user_template(username, template)
        templates[username] = template.tolist()
        update_user_list()

    else:
        messagebox.showwarning("Warning", "Please enter a username.")


def exit_app(root: tk.Tk) -> None:
    global running, cap

    cap.release()
    cv.destroyAllWindows()
    running = False
    root.destroy()


def update_video_feed() -> None:
    global face, running, cap

    if not running:
        return

    face, frame = capture_and_extract_face(
        cap=cap,
        final_size=(480, 640),
        min_neighbors=MIN_NEIGHBORS,
        scale_factor=SCALE_FACTOR,
    )

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)
    video_label.configure(image=frame)
    video_label.image = frame

    video_label.after(REFRESH_TIME, update_video_feed)


def update_user_list() -> None:
    global face, templates, users_list

    if face is None:
        messagebox.showwarning("Warning", "No face detected")
        return

    new_template = gen_template(face)

    # delete all entries in listbox
    users_list.delete(0, tk.END)

    for username, template in templates.items():
        distance = compare_bloom_filters(new_template, template)

        # add distance to listbox for each user and remove the previous entry
        users_list.insert(tk.END, f"{username} - {distance}")


def main() -> None:
    global users_list, templates, video_label

    root = tk.Tk()
    root.title("Face Registration App")
    root.protocol("WM_DELETE_WINDOW", exit_app)

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    video_frame = tk.LabelFrame(root, text="Video Feed", width=640, height=480)
    video_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky=tk.NSEW)
    video_frame.pack_propagate(False)

    video_label = tk.Label(
        video_frame, text="Video feed will be displayed here", bg="grey"
    )
    video_label.pack(expand=True, fill=tk.BOTH)

    username_label = tk.Label(root, text="Username:")
    username_label.grid(row=1, column=0, sticky=tk.W, padx=10)
    username_entry = tk.Entry(root)
    username_entry.grid(row=1, column=1, sticky=tk.EW, padx=10)

    users_list = tk.Listbox(root)
    users_list.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=10)

    register_func = partial(register_user, username_entry, users_list)
    register_button = tk.Button(root, text="Register User", command=register_func)
    register_button.grid(row=2, column=0, padx=10, pady=10, sticky=tk.EW)

    exit_func = partial(exit_app, root)
    exit_button = tk.Button(root, text="Exit", command=exit_func)
    exit_button.grid(row=2, column=1, padx=10, pady=10, sticky=tk.NSEW)

    # add button to compare image to database
    find_button = tk.Button(root, text="Find User", command=update_user_list)
    find_button.grid(row=4, column=0, padx=10, pady=10, sticky=tk.EW)

    global running, cap, face, templates

    running = True
    face = None
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r") as file:
            templates = json.load(file)
        for username in templates:
            users_list.insert(tk.END, username)
    else:
        templates = {}

    update_video_feed()
    root.mainloop()


if __name__ == "__main__":
    main()
