#!/usr/bin/env bash
rm *.log

log='pipeline.log'
exec 2>>$log

python src/pipeline.py
