#!/bin/bash
# Script to setup PostgreSQL database and user

# Create user 'xx' with no password
sudo -u postgres psql <<EOF
CREATE USER x;
ALTER USER x WITH SUPERUSER;
ALTER USER x WITH PASSWORD '123';
CREATE DATABASE rbacdatabase_treebase OWNER x;
\c rbacdatabase_treebase
CREATE EXTENSION IF NOT EXISTS vector;
EOF

echo "Database setup completed!"

# go to database command:
# sudo -u postgres psql rbacdatabase_treebase

# close and open PostgreSQL service:
# sudo systemctl stop postgresql
# sudo systemctl start postgresql