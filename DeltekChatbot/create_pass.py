import sqlite3
import bcrypt

db_path = "users.db"
admin_username = "Admin1"
password = "Password1"
hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

conn = sqlite3.connect(db_path)
c = conn.cursor()
try:
    c.execute('INSERT INTO users (username, password, access) VALUES (?, ?, ?)', (admin_username, hashed_password, 1))
    conn.commit()
    print("Admin user created successfully.")
except sqlite3.IntegrityError:
    print("Username already exists.")
conn.close()
