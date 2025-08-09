from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 修复CORS - 添加你的实际Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sentifi-portfolio.vercel.app",  # 你的实际Vercel URL！
        "https://sentifi-portfolio-*.vercel.app",  # 预览部署
        "http://localhost:3000",  # 本地开发
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有headers
)