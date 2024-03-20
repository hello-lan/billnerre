#!/bin/bash
# Author:LHQ
# Create Time:Thu Mar 14 11:09:10 2024
nohup uvicorn api:app  --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
