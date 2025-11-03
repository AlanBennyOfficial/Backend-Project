
import mysql.connector
import os

# --- Database Configuration ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""  # --- IMPORTANT: Replace with your MySQL root password ---
DB_NAME = "nipe-attendance"

# --- Dataset Path ---
DATASET_PATH = "dataset"

def create_database():
    """Creates the database and tables."""
    try:
        # Connect to MySQL server
        cnx = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = cnx.cursor()

        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
        print(f"Database '{DB_NAME}' created or already exists.")

        # Use the database
        cursor.execute(f"USE `{DB_NAME}`")

        # Create students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `students` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `name` VARCHAR(255) NOT NULL,
                `email` VARCHAR(255)
            )
        """)
        print("Table 'students' created or already exists.")

        # Create attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `attendance` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `student_id` INT NOT NULL,
                `status` VARCHAR(50) NOT NULL DEFAULT 'Absent',
                `date` DATE NOT NULL,
                FOREIGN KEY (`student_id`) REFERENCES `students`(`id`)
            )
        """)
        print("Table 'attendance' created or already exists.")

        # Populate students table
        populate_students(cursor)

        cnx.commit()
        cursor.close()
        cnx.close()
        print("Database setup complete.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

def populate_students(cursor):
    """Populates the students table from the dataset directory."""
    student_names = [name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))]

    for name in student_names:
        display_name = name.replace("-", " ") # Convert 'firstname-lastname' to 'firstname lastname'
        # Check if student already exists
        cursor.execute("SELECT id FROM students WHERE name = %s", (display_name,))
        if cursor.fetchone() is None:
            cursor.execute("INSERT INTO students (name) VALUES (%s)", (display_name,))
            print(f"Inserted student: {display_name}")

if __name__ == "__main__":
    create_database()
