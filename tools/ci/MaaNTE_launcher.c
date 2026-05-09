#include <windows.h>
#include <shellapi.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);

    char *last_slash = strrchr(path, '\\');
    if (last_slash) *(last_slash + 1) = '\0';

    char app[MAX_PATH];
    snprintf(app, MAX_PATH, "%sMaaNTE-app.exe", path);

    if (!IsUserAnAdmin()) {
        int result = MessageBoxA(
            NULL,
            "当前未以管理员权限运行，部分功能可能无法正常使用。\n\n是否立即以管理员身份重新启动？",
            "MaaNTE - 权限不足",
            MB_YESNO | MB_ICONWARNING
        );

        if (result == IDYES) {
            HINSTANCE ret = ShellExecuteA(
                NULL, "runas", app, NULL, NULL, SW_SHOWNORMAL
            );
            if ((intptr_t)ret > 32) return 0;
        }
    }

    STARTUPINFOA si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    if (CreateProcessA(app, NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    return 0;
}
