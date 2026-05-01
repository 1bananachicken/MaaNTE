// MaaNTE 首次使用警告弹窗启动器
// 编译: cl /O2 /Fe:MaaNTE.exe launcher.c /link /SUBSYSTEM:WINDOWS

#include <windows.h>
#include <stdio.h>

#define CONFIG_PATH L"config\\warning_shown.json"
#define CORE_EXE    L"MaaNTE_core.exe"
#define TIMEOUT_MS  5000

static const WCHAR *WARN_TITLE = L"MaaNTE - 首次使用须知";
static const WCHAR *WARN_TEXT =
    L"欢迎使用 MaaNTE\r\n\r\n"
    L"MaaNTE 为免费开源项目，从未授权任何人以任何形式进行售卖。\r\n"
    L"  - 如在闲鱼、淘宝等平台购买了本软件，请立即申请退款并举报商家\r\n"
    L"  - 可凭此弹窗截图要求退款，维护自身权益\r\n"
    L"  - 你付给倒卖者的每一分钱都会让开源社区更艰难\r\n\r\n"
    L"Mirror酱 是我们的合作伙伴，提供下载加速服务，不属于售卖行为\r\n\r\n"
    L"───────────────────────────\r\n\r\n"
    L"本软件开源免费，仅供学习交流使用。\r\n"
    L"使用本软件产生的所有后果由使用者自行承担，与开发者团队无关。\r\n"
    L"开发者团队拥有本项目的最终解释权。";

static int is_warning_shown(void) {
    HANDLE h = CreateFileW(CONFIG_PATH, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;

    char buf[256] = {0};
    DWORD read;
    ReadFile(h, buf, sizeof(buf) - 1, &read, NULL);
    CloseHandle(h);
    return (strstr(buf, "\"shown\": true") != NULL ||
            strstr(buf, "\"shown\":true") != NULL);
}

static void mark_shown(void) {
    CreateDirectoryW(L"config", NULL);
    HANDLE h = CreateFileW(CONFIG_PATH, GENERIC_WRITE, 0, NULL,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return;
    const char *data = "{\"shown\": true}";
    DWORD written;
    WriteFile(h, data, (DWORD)strlen(data), &written, NULL);
    CloseHandle(h);
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR cmd, int show) {
    if (!is_warning_shown()) {
        HWND hwnd = GetForegroundWindow();
        typedef int (WINAPI *MsgBoxTimeoutW)(HWND, LPCWSTR, LPCWSTR, UINT, WORD, DWORD);
        MsgBoxTimeoutW pMsgBox = (MsgBoxTimeoutW)GetProcAddress(
            GetModuleHandleW(L"user32.dll"), "MessageBoxTimeoutW");

        if (pMsgBox) {
            pMsgBox(hwnd, WARN_TEXT, WARN_TITLE,
                    MB_OK | MB_TOPMOST | MB_ICONWARNING | MB_SETFOREGROUND,
                    0, TIMEOUT_MS);
        } else {
            MessageBoxW(hwnd, WARN_TEXT, WARN_TITLE,
                        MB_OK | MB_TOPMOST | MB_ICONWARNING);
        }
        mark_shown();
    }

    WCHAR path[MAX_PATH];
    GetModuleFileNameW(NULL, path, MAX_PATH);
    WCHAR *slash = wcsrchr(path, L'\\');
    if (slash) *(slash + 1) = L'\0';
    wcscat_s(path, MAX_PATH, CORE_EXE);

    STARTUPINFOW si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    CreateProcessW(path, NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);

    if (pi.hProcess) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    return 0;
}
