#include <iostream>

#include <MaaAgentServer/MaaAgentServerAPI.h>
#include <MaaFramework/MaaAPI.h>
#include <MaaUtils/Logger.h>

#include "actions.h"
#include "util.h"

int main(int argc, char** argv)
{
    navi::setup_runtime_dll_search_path();
    if (argc < 2) {
        std::cerr << "Usage: cpp-navi <identifier>" << std::endl;
        return 2;
    }

    navi::start_parent_process_watcher();
    const std::string log_dir = (navi::project_root() / "debug" / "cpp-navi").string();
    if (!MaaGlobalSetOption(MaaGlobalOption_LogDir, const_cast<char*>(log_dir.data()), log_dir.size())) {
        std::cerr << "Failed to configure MaaFramework log directory" << std::endl;
    }
    auto& logger = MaaNS::LogNS::Logger::get_instance();
    logger.start_logging(log_dir);
    logger.set_stdout_level(MaaNS::LogNS::level::info);
    LogInfo << "C++ Navi Agent starting" << VAR(log_dir);

    if (!MaaAgentServerRegisterCustomAction("check_teleport_required", navi::check_teleport_required, nullptr)
        || !MaaAgentServerRegisterCustomAction("online_map_navigation", navi::online_map_navigation, nullptr)
        || !MaaAgentServerRegisterCustomAction("local_route_navigation", navi::local_route_navigation, nullptr)
        || !MaaAgentServerRegisterCustomAction("local_route_navigation_unit_test", navi::local_route_navigation_unit_test, nullptr)) {
        std::cerr << "Failed to register Navi custom actions" << std::endl;
        return 3;
    }

    const char* identifier = argv[argc - 1];
    if (!MaaAgentServerStartUp(identifier)) {
        std::cerr << "Failed to start Navi AgentServer" << std::endl;
        return 4;
    }
    MaaAgentServerJoin();
    MaaAgentServerShutDown();
    return 0;
}
