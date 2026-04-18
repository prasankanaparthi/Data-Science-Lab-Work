import tkinter as tk
import random

# Word list
words = ["apple", "grape", "mango", "peach", "lemon"]
secret_word = random.choice(words)

attempt = 0
max_attempts = 6

# Colors
GREEN = "#6aaa64"
YELLOW = "#c9b458"
GRAY = "#787c7e"
WHITE = "#ffffff"

# Create window
root = tk.Tk()
root.title("Wordle Game")
root.geometry("300x400")
root.config(bg="black")

# Grid for letters
labels = []
for i in range(6):
    row = []
    for j in range(5):
        label = tk.Label(root, text="", width=4, height=2,
                         font=("Helvetica", 18, "bold"),
                         bg="black", fg="white", relief="solid", bd=1)
        label.grid(row=i, column=j, padx=3, pady=3)
        row.append(label)
    labels.append(row)

# Entry box
entry = tk.Entry(root, font=("Helvetica", 18), justify="center")
entry.grid(row=7, column=0, columnspan=5, pady=10)

def check_word():
    global attempt
    
    guess = entry.get().lower()
    
    if len(guess) != 5:
        return
    
    for i in range(5):
        if guess[i] == secret_word[i]:
            labels[attempt][i].config(text=guess[i].upper(), bg=GREEN)
        elif guess[i] in secret_word:
            labels[attempt][i].config(text=guess[i].upper(), bg=YELLOW)
        else:
            labels[attempt][i].config(text=guess[i].upper(), bg=GRAY)
    
    attempt += 1
    entry.delete(0, tk.END)

    # Win
    if guess == secret_word:
        entry.insert(0, "You Win!")
        entry.config(state="disabled")
    
    # Lose
    elif attempt == max_attempts:
        entry.insert(0, f"Word was: {secret_word}")
        entry.config(state="disabled")

# Button
button = tk.Button(root, text="Submit", command=check_word)
button.grid(row=8, column=0, columnspan=5)

root.mainloop()