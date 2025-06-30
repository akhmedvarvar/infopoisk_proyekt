from api.endpoints import router
from fastapi import FastAPI

app = FastAPI()

# Подключаем роуты
app.include_router(router)
