import os


os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session
)

import tensorflow as tf
import numpy as np
import cv2
import sqlite3

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


app = Flask(__name__)

app.secret_key = "chicken-disease-secret-key"

DATABASE = "database.db"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)





# ---------------- DATABASE ----------------

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            farm_name TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_name TEXT,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            date_time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------- MODEL ----------------

model = tf.keras.models.load_model(
    "model/efficientnetb3-Chicken Disease-98.14.h5",
    compile=False
)


classes = [
    "Coccidiosis",
    "Healthy",
    "New Castle Disease",
    "Salmonella"
]


# ---------------- HOME ----------------

@app.route("/")
def home():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip()
        password = request.form["password"]

        conn = get_db()

        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid email or password."
        )

    return render_template("login.html")


# ---------------- REGISTER ----------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        farm_name = request.form["farm"].strip()

        hashed_password = generate_password_hash(password)

        conn = get_db()

        try:

            conn.execute(
                """
                INSERT INTO users
                (name, email, password, farm_name)
                VALUES (?, ?, ?, ?)
                """,
                (
                    name,
                    email,
                    hashed_password,
                    farm_name
                )
            )

            conn.commit()
            conn.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            conn.close()

            return render_template(
                "register.html",
                error="Email already registered."
            )

    return render_template("register.html")


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        user_name=session.get("user_name")
    )


# ---------------- PREDICTION ----------------

@app.route("/predict", methods=["POST"])
def predict():

    if "user_id" not in session:

        return jsonify({
            "error": "Please login first."
        }), 401

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected"
        }), 400

    filename = secure_filename(file.filename)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    saved_filename = timestamp + "_" + filename

    file_path = os.path.join(
        UPLOAD_FOLDER,
        saved_filename
    )

    file.save(file_path)

    image = cv2.imread(file_path)

    if image is None:

        return jsonify({
            "error": "Invalid image"
        }), 400

    image = cv2.resize(image, (224, 224))

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )

    class_index = np.argmax(prediction[0])

    confidence = float(
        prediction[0][class_index]
    )

    disease = classes[class_index]

    conn = get_db()

    conn.execute(
        """
        INSERT INTO predictions
        (user_id, image_name, disease, confidence, date_time)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session["user_id"],
            saved_filename,
            disease,
            confidence,
            datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
        )
    )

    conn.commit()
    conn.close()

    return jsonify({
        "disease": disease,
        "confidence": confidence
    })


# ---------------- HISTORY ----------------

@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# ---------------- DISEASE INFORMATION ----------------

@app.route("/disease-info")
def disease_info():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("disease_info.html")


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)