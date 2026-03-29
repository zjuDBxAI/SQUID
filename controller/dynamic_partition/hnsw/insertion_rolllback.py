import psycopg2
from psycopg2 import sql
from services.config import get_db_connection

def rollback_database():
    """
    Rollback the database to the backup state by truncating tables and restoring from backup tables.
    Ensures that all foreign key dependencies are handled with CASCADE.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        print("[INFO] Rolling back database to previous backup state...")

        # Step 1: Clear tables with CASCADE to handle foreign key dependencies
        cur.execute("TRUNCATE TABLE Users CASCADE;")
        cur.execute("TRUNCATE TABLE Roles CASCADE;")
        cur.execute("TRUNCATE TABLE UserRoles CASCADE;")
        cur.execute("TRUNCATE TABLE PermissionAssignment;")
        cur.execute("TRUNCATE TABLE combrolepartitions CASCADE;")

        # Step 2: Restore data from backup tables
        cur.execute("INSERT INTO Users SELECT * FROM Users_backup;")
        cur.execute("INSERT INTO Roles SELECT * FROM Roles_backup;")
        cur.execute("INSERT INTO UserRoles SELECT * FROM UserRoles_backup;")
        cur.execute("INSERT INTO PermissionAssignment SELECT * FROM PermissionAssignment_backup;")
        cur.execute("INSERT INTO combrolepartitions SELECT * FROM combrolepartitions_backup;")

        conn.commit()
        print("[SUCCESS] Database rollback completed successfully.")

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to rollback database: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    rollback_database()
