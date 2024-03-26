//
// Created by Robert Sowula on 27.06.23.
//

#include "key_utils.h"

KeyFinder::AudioData loadAudioDataFromPCM(const keyfinder::KeyRequest* request) {
    spdlog::debug("Loading AudioData from in-memory PCM data");

    // we assume that the PCM data is in 32-bit float format as most platforms should adhere to the IEEE 754 standard
    // since float32_t is not yet implemented in clang, we opted to use this assert
    static_assert(sizeof(float) * CHAR_BIT == 32, "float is not 32 bits on this platform");

    std::vector<float> pcmData;
    pcmData.resize(request->pcm_data().size() / sizeof(float));
    std::memcpy(pcmData.data(), request->pcm_data().data(), request->pcm_data().size());

    spdlog::trace("Size of pcm data: {} bytes", pcmData.size() * sizeof(float));

    // Create an AudioData object and fill it with the samples
    KeyFinder::AudioData audioData;
    audioData.setFrameRate(request->frame_rate());
    audioData.setChannels(request->channels());
    audioData.addToSampleCount(pcmData.size());
    for (int i = 0; i < pcmData.size(); i++) {
        audioData.setSample(i, pcmData[i]);
    }


    return audioData;
}

std::unordered_map<KeyFinder::key_t, std::string> keyToCamelotMap = {
        {KeyFinder::A_MAJOR, "11B"},
        {KeyFinder::A_MINOR, "8A"},
        {KeyFinder::B_FLAT_MAJOR, "6B"},
        {KeyFinder::B_FLAT_MINOR, "3A"},
        {KeyFinder::B_MAJOR, "1B"},
        {KeyFinder::B_MINOR, "10A"},
        {KeyFinder::C_MAJOR, "8B"},
        {KeyFinder::C_MINOR, "5A"},
        {KeyFinder::D_FLAT_MAJOR, "3B"},
        {KeyFinder::D_FLAT_MINOR, "12A"},
        {KeyFinder::D_MAJOR, "10B"},
        {KeyFinder::D_MINOR, "7A"},
        {KeyFinder::E_FLAT_MAJOR, "5B"},
        {KeyFinder::E_FLAT_MINOR, "2A"},
        {KeyFinder::E_MAJOR, "12B"},
        {KeyFinder::E_MINOR, "9A"},
        {KeyFinder::F_MAJOR, "7B"},
        {KeyFinder::F_MINOR, "4A"},
        {KeyFinder::G_FLAT_MAJOR, "2B"},
        {KeyFinder::G_FLAT_MINOR, "11A"},
        {KeyFinder::G_MAJOR, "9B"},
        {KeyFinder::G_MINOR, "6A"},
        {KeyFinder::A_FLAT_MAJOR, "4B"},
        {KeyFinder::A_FLAT_MINOR, "1A"},
        {KeyFinder::SILENCE, ""}
};

std::string keyToCamelot(KeyFinder::key_t key) {
    spdlog::debug("Converting key to camelot notation");
    auto it = keyToCamelotMap.find(key);
    if (it != keyToCamelotMap.end()) {
        return it->second;
    }
    return "Unknown";
}