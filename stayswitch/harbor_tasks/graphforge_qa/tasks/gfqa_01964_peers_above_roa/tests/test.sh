#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify.py /tests/expected.json /app/answer.txt /logs/verifier/reward.txt
