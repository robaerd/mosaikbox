//
// Created by Robert Sowula on 27.06.23.
//

#ifndef KEYFINDERSERVICE_KEY_UTILS_H
#define KEYFINDERSERVICE_KEY_UTILS_H

#include <vector>
#include <string>
#include "keyfinder.h"
#include "keyfinder.pb.h"
#include "keyfinder.grpc.pb.h"
#include "spdlog/spdlog.h"

std::string keyToCamelot(KeyFinder::key_t key);

KeyFinder::AudioData loadAudioDataFromPCM(const keyfinder::KeyRequest* request);


#endif //KEYFINDERSERVICE_KEY_UTILS_H
