#!/bin/sh

python3 -m grpc_tools.protoc -I ../../KeyFinderService/proto --python_out=. --grpc_python_out=. ../../KeyFinderService/proto/keyfinder.proto
