-- Create the application user with necessary permissions
CREATE USER app_user WITH PASSWORD 'app_password';
ALTER USER app_user CREATEDB;

CREATE DATABASE app_database OWNER app_user;
GRANT ALL PRIVILEGES ON DATABASE app_database TO app_user;

-- Connect to the application database and create the vector extension
\c app_database
CREATE EXTENSION IF NOT EXISTS vector;