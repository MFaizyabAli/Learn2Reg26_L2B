#!/usr/bin/env bash
echo "Building docker image..."
docker build -f Dockerfile -t learn2breath_convexadam .
