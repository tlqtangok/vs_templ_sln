#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
int main() {

json ex1 = json::parse(R"(
  {
    "pi": 3.141,
    "happy": true
  }
)");


    std::cout << "=== Test 1: Basic JSON Parsing ===" << std::endl;
    std::cout << "Parsed JSON:" << std::endl;
    std::cout << ex1.dump(4) << std::endl;

    std::cout << "=== All tests completed successfully! ===" << std::endl;
    
    return 0;
}
