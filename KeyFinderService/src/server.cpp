//
// Created by Robert Sowula on 27.06.23.
//

#include "server.h"
#include <grpcpp/grpcpp.h>
#include "keyfinder.pb.h"
#include "keyfinder.grpc.pb.h"
#include "spdlog/spdlog.h"
#include "keyfinder.h"  // include the libkeyfinder
#include "key_utils.h"
#include "spdlog/sinks/stdout_color_sinks.h"

class KeyFinderServiceImpl final : public keyfinder::KeyFinder::Service {
    grpc::Status GetKey(grpc::ServerContext* context, const keyfinder::KeyRequest* request, keyfinder::KeyResponse* response) override {
        spdlog::info("Received a GetKey request");
        // Convert the bytes to audio data
        KeyFinder::AudioData audio_data = loadAudioDataFromPCM(request);

        // Call libkeyfinder to get the key
        KeyFinder::KeyFinder kf;

        auto key = kf.keyOfAudio(audio_data);

        // Convert the key to a string and set the response
        std::string keyCamelot = keyToCamelot(key);

        response->set_key(keyCamelot);

        return grpc::Status::OK;
    }
};

void RunServer() {
    spdlog::debug("Starting gRPC server");
    std::string server_address("0.0.0.0:50051");
    KeyFinderServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Set the maximum receive message size (here we're setting it to 2GB)
    builder.SetMaxReceiveMessageSize(2000 * 1024 * 1024);

    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    server->Wait();
}

void configureLogger() {
    // Set the default logger to output to stdout with color
    spdlog::set_default_logger(spdlog::stdout_color_mt("default"));
    // Set the log level
    spdlog::set_level(spdlog::level::trace);
    spdlog::debug("Configured logger");
}

int main(int argc, char** argv) {
    configureLogger();
    RunServer();
    return 0;
}
