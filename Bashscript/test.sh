#!/usr/bin/env bash

until python test.py
do
    echo "Restarting"
    sleep 2
done