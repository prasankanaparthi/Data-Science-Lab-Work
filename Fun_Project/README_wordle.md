# Wordle Game

## Description
A desktop clone of the popular Wordle word-guessing game built with Python's Tkinter GUI library. The player has 6 attempts to guess a hidden 5-letter word, with colour-coded feedback after each guess.

## Dataset
- **Source:** Hardcoded list
- **Description:** A small set of 5-letter fruit words (`apple`, `grape`, `mango`, `peach`, `lemon`). One word is randomly selected as the secret word each game.

## Steps Performed
1. **Data Cleaning** — N/A (static word list)
2. **Exploratory Data Analysis** — N/A
3. **Visualization** — Tkinter 6×5 grid of letter tiles with colour feedback: 🟩 Green = correct position, 🟨 Yellow = wrong position, ⬜ Gray = not in word
4. **Model Building** — N/A (rule-based game logic)

## Results
- **Win condition:** Guess the exact word within 6 attempts → displays "You Win!"
- **Lose condition:** Exhausted all 6 attempts → reveals the secret word
- Input validation ensures only 5-character guesses are processed

## Tools Used
- Python
- Tkinter (built-in GUI library)
- `random` (standard library)

## Conclusion
A lightweight, playable Wordle implementation demonstrating event-driven GUI programming in Python. Can be extended with a larger word dictionary, keyboard widget, and hard-mode constraints.

## Author
Prasan Kanaparthi
