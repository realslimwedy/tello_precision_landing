import tkinter as tk

def on_key_press(window, event):
    key = event.keysym
    if key == 'Escape':
        print("ESC key pressed: Exiting the application...")
        window.destroy()
    else:
        print(f"Key pressed: {key}")

def run_application():
    window = tk.Tk()
    window.bind("<KeyPress>", lambda event: on_key_press(window, event))
    window.mainloop()
