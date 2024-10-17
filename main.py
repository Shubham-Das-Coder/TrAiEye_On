import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image, ImageTk

def paste_image():
    """Retrieve the image from clipboard and display it."""
    try:
        img = ImageGrab.grabclipboard()  # Get the clipboard image
        if img is None:
            raise ValueError("No image found in clipboard.")

        # Create a new window to display the image
        display_window = tk.Toplevel()
        display_window.title("Displayed Image")

        # Load the image and display it
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(display_window, image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack()

    except ValueError as ve:
        messagebox.showwarning("No Image", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Set up the main application window
root = tk.Tk()
root.title("Paste Image to Display")
root.geometry("300x150")

# Create a button to paste the image
paste_button = tk.Button(root, text="Paste Image", command=paste_image)
paste_button.pack(pady=50)

# Start the GUI event loop
root.mainloop()
