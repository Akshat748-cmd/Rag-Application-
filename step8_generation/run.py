import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8007,           # Step 8 runs on port 8007
        reload=True,
        reload_dirs=[os.path.dirname(__file__)]
    )
