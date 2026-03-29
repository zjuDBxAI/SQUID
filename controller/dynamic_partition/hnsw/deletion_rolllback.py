import psycopg2
from psycopg2 import sql
from services.config import get_db_connection

def backup_database():
    """
    Backup critical tables by copying their contents into backup tables.
    Ensures that if rollback is needed, we can restore the original state.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        print("[INFO] Creating backup of the database...")

        # Step 1: Drop old backup tables if they exist
        cur.execute("DROP TABLE IF EXISTS Users_backup CASCADE;")
        cur.execute("DROP TABLE IF EXISTS Roles_backup CASCADE;")
        cur.execute("DROP TABLE IF EXISTS UserRoles_backup CASCADE;")
        cur.execute("DROP TABLE IF EXISTS PermissionAssignment_backup CASCADE;")
        cur.execute("DROP TABLE IF EXISTS combrolepartitions_backup CASCADE;")

        # Step 2: Create new backup tables
        cur.execute("CREATE TABLE Users_backup AS TABLE Users;")
        cur.execute("CREATE TABLE Roles_backup AS TABLE Roles;")
        cur.execute("CREATE TABLE UserRoles_backup AS TABLE UserRoles;")
        cur.execute("CREATE TABLE PermissionAssignment_backup AS TABLE PermissionAssignment;")
        cur.execute("CREATE TABLE combrolepartitions_backup AS TABLE combrolepartitions;")

        conn.commit()
        print("[SUCCESS] Database backup completed successfully.")

    except psycopg2.Error as e:
        print(f"[ERROR] Failed to backup database: {e}")
        conn.rollback()

    finally:
        cur.close()
        conn.close()


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
    # print("1. **Backing up database before modification**")
    # backup_database()

    # Perform deletion or update operations here...

    # If rollback is needed
    print("2. **Rolling back database to previous state**")
    rollback_database()
