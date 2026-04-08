"""from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "mysql+aiomysql://%40@localhost/character_prediction_model"

engine = create_async_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


Base = declarative_base()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)"""


import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. Get your actual credentials from environment variables
db_user = os.getenv("DB_USER", "mysql")        # Replace 'postgres' with your default user
db_pass = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")      # Replace with your actual database host
db_port = os.getenv("DB_PORT", "8000")           # Replace with your actual port
db_name = os.getenv("DB_NAME", "character_prediction_model")     # Replace with your actual database name

# 2. Add '+asyncpg' to the URL (required for create_async_engine)
# 3. Use the variables in the f-string
DATABASE_URL = f"mysql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

engine = create_async_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
