import uvicorn, os, sys
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,           # Step 2 runs on 8001 (Step 1 is on 8000)
        reload=True,
        reload_dirs=[os.path.dirname(__file__)]
    )
