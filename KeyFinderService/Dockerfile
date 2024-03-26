# Stage 1: Build Stage
FROM fedora:39 AS builder

# Update and install necessary packages
RUN dnf update -y && dnf install -y \
    make automake kernel-devel \
    autoconf \
    libtool \
    pkg-config \
    cmake \
    git \
    clang \
    curl \
    wget \
    unzip

# Set clang as default C and C++ compiler
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

ENV MY_INSTALL_DIR=/usr/local

WORKDIR /grpc
RUN git clone --recurse-submodules -b v1.61.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc .
RUN mkdir -p cmake/build && \
    cd cmake/build && \
    cmake -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_TESTS=OFF \
          -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
          ../.. && \
    make -j $(nproc) && \
    make install

WORKDIR /compile/fftw3
RUN wget http://fftw.org/fftw-3.3.10.tar.gz
RUN tar -xvf fftw-3.3.10.tar.gz
WORKDIR /compile/fftw3/fftw-3.3.10
RUN mkdir build && cd build && \
    cmake -DBUILD_SHARED_LIBS=OFF .. && \
    make -j $(nproc) && \
    make install


WORKDIR /compile/libkeyfinder
RUN git clone https://github.com/robaerd/libkeyfinder.git .
RUN mkdir -p /libkeyfinder
RUN cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/libkeyfinder -DBUILD_TESTING=OFF -S . -B build
RUN cmake --build build --parallel $(nproc)
RUN cmake --install build

ADD . /app
WORKDIR /app
RUN cmake -DLIB_SUFFIX="64" -DSHOULD_INCLUDE_FFTW3=ON -DKEYFINDER_PATH="/libkeyfinder" -S . -B build
RUN cmake --build build --parallel $(nproc)

# Stage 2: Server Stage
FROM fedora:39 AS server

# Copy only the built binary from the previous stage
COPY --from=builder /app/build/server /app/

# Run your server
WORKDIR /app
CMD ["./server"]
