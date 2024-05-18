#include <catch2/catch_test_macros.hpp>
#include "parse.hpp"
#include "yaml-cpp/yaml.h" //https://github.com/jbeder/yaml-cpp

TEST_CASE("parse_yaml", "test") {
    REQUIRE(parse::get_len_yaml(YAML::Load("[1, 2, 3]")) == 3); 
    REQUIRE(parse::get_val_yaml(YAML::Load("{key1: TEST1, key2: TEST2, key3: TEST3}"), "key3") == "TEST3");
}

TEST_CASE("parse_yaml2", "test"){
    REQUIRE(2 * 2 == 4);
    // REQUIRE(2 * 2 == 8);
}
