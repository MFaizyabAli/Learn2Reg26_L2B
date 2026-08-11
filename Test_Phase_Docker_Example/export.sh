#!/usr/bin/env bash

bash ./build.sh

docker save learn2breath_convexadam | gzip -c > learn2breath_convexadam.tar.gz
