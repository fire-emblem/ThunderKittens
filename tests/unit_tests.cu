#include "testing_flags.cuh"
#include "testing_commons.cuh"

#ifdef TEST_THREAD
#include "thread/thread.cuh"
#endif
#ifdef TEST_GROUP
#include "group/group.cuh"
#endif

#ifdef TEST_C500
#include "c500/c500.cuh"
#endif

namespace {

bool is_pending_backend_scaffold(const test_info &info) {
    return info.result == test_result::INVALID &&
           info.label.find("[pending_backend]") != std::string::npos;
}

} // namespace

int main(int argc, char **argv) {

    should_write_outputs = argc>1; // write outputs if user says so

    test_data data;

#ifdef TEST_THREAD
    thread::tests(data);
#else
    std::cout << "INFO: Skipping ops/thread tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP
    group::tests(data);
#else
    std::cout << "INFO: Skipping ops/group tests!\n" << std::endl;
#endif
#ifdef TEST_C500
    c500::tests(data);
#else
    std::cout << "INFO: Skipping ops/c500 tests!\n" << std::endl;
#endif

    std::cout << " ------------------------------     Summary     ------------------------------\n"  << std::endl;

    std::cout << "Failed tests:\n";
    int passes = 0, fails = 0, invalids = 0, pending_backend = 0;
    for(auto it = data.begin(); it != data.end(); it++) {
        if(it->result == test_result::PASSED)  passes++;
        if(it->result == test_result::INVALID) invalids++;
        if(is_pending_backend_scaffold(*it)) pending_backend++;
        if(it->result == test_result::FAILED) {
            fails++;
            std::cout << it->label << std::endl;
        }
    }
    if(fails == 0) {
        if(pending_backend > 0) {
            std::cout << "NO TEST FAILURES, BUT " << pending_backend
                      << " C500 probe scaffolds are still pending backend implementation.\n";
        }
        else {
            std::cout << "ALL TESTS PASSED!\n";
        }
    }
    if(pending_backend > 0) {
        std::cout << "\nPending backend scaffold probes:\n";
        for(auto it = data.begin(); it != data.end(); it++) {
            if(is_pending_backend_scaffold(*it)) {
                std::cout << it->label << std::endl;
            }
        }
    }
    std::cout << std::endl;

    std::cout << invalids - pending_backend << " test template configurations deemed invalid (this is normal)\n";
    if(pending_backend > 0) {
        std::cout << pending_backend << " scaffold probes pending backend implementation\n";
    }
    std::cout << passes   << " tests passed\n";
    std::cout << fails    << " tests failed\n";

    return 0;
}
