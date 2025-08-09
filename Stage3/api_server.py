from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS配置 - 确保包含你的Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sentifi-portfolio.vercel.app",  # 你的Vercel URL
        "http://localhost:3000",
        "*"  # 临时测试，之后删除
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SentiFi API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/supported_assets")  # 下划线版本！
async def get_supported_assets():
    return {
        "assets": ["BTC", "ETH", "SOL", "ADA", "MATIC", "DOT", "LINK", "AVAX", "UNI", "ATOM"]
    }