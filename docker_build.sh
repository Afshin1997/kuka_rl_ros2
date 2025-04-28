#!/bin/bash
docker build -t catch_throw --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
