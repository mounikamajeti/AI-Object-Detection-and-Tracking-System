import customtkinter as ctk
from gui.dashboard import Dashboard

if __name__ == '__main__':
    app = ctk.CTk()
    Dashboard(app)
    app.mainloop()
