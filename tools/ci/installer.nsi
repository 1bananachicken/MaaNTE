!include "MUI2.nsh"
!include "LogicLib.nsh"

; ---------- 动态参数（可在命令行通过 /D 指定） ----------
!ifndef VERSION
  !define VERSION "1.0.0"
!endif
!ifndef ARCH
  !define ARCH "x64"
!endif
!ifndef ICON_FILE
  !define ICON_FILE "H:\MaaNTE\logo.ico"
!endif
!ifndef SOURCE_DIR
  !define SOURCE_DIR "H:\MaaNTE"
!endif

; ---------- 基本信息 ----------
Name "MaaNTE"
!ifndef OUTFILE
  !define OUTFILE "MaaNTE_Setup.exe"
!endif
OutFile "${OUTFILE}"
InstallDir "D:\MaaNTE"
InstallDirRegKey HKLM "Software\MaaNTE" "InstallDir"
RequestExecutionLevel admin
SetCompressor /SOLID zstd

; ---------- 图标（使用动态变量） ----------
!define MUI_ICON "${ICON_FILE}"
!define MUI_UNICON "${ICON_FILE}"

; ---------- 页面 ----------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "SimpChinese"

; ---------- 组件描述 ----------
LangString DESC_SecCore ${LANG_SIMPCHINESE} "核心程序文件（必选）"
LangString DESC_SecStartMenu ${LANG_SIMPCHINESE} "在开始菜单中创建 MaaNTE 的快捷方式"
LangString DESC_SecDesktop ${LANG_SIMPCHINESE} "在桌面上创建 MaaNTE 的快捷方式"

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} $(DESC_SecCore)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} $(DESC_SecStartMenu)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} $(DESC_SecDesktop)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ---------- 安装区段 ----------
Section "核心文件" SecCore
    SectionIn RO
    SetOutPath "$INSTDIR"

    ; 安装/更新前清理旧文件夹，避免残留文件干扰
    ClearErrors
    RMDir /r "$INSTDIR\agent"
    RMDir /r "$INSTDIR\cache"
    RMDir /r "$INSTDIR\resource"
    RMDir /r "$INSTDIR\maafw"
    RMDir /r "$INSTDIR\python"

    ; 使用动态源文件目录
    File /r "${SOURCE_DIR}\*.*"

    ; 创建卸载程序
    WriteUninstaller "$INSTDIR\uninstall.exe"

    ; 注册表（程序和功能）
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "DisplayName" "MaaNTE"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "UninstallString" "$INSTDIR\uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "DisplayIcon" "$INSTDIR\logo.ico"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "Publisher" "MaaNTE"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "InstallLocation" "$INSTDIR"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "NoRepair" 1
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE" "DisplayVersion" "${VERSION}"

    WriteRegStr HKLM "Software\MaaNTE" "InstallDir" "$INSTDIR"
SectionEnd

Section /o "开始菜单快捷方式" SecStartMenu
    SetShellVarContext all
    CreateDirectory "$SMPROGRAMS\MaaNTE"
    CreateShortCut "$SMPROGRAMS\MaaNTE\MaaNTE.lnk" "$INSTDIR\MaaNTE.exe" "" "$INSTDIR\logo.ico" 0
    CreateShortCut "$SMPROGRAMS\MaaNTE\Uninstall.lnk" "$INSTDIR\uninstall.exe" "" "$INSTDIR\logo.ico" 0
SectionEnd

Section /o "桌面快捷方式" SecDesktop
    SetShellVarContext all
    CreateShortCut "$DESKTOP\MaaNTE.lnk" "$INSTDIR\MaaNTE.exe" "" "$INSTDIR\logo.ico" 0
SectionEnd

; ---------- .onInit（组件默认选中） ----------
Function .onInit
    ; 默认选中开始菜单和桌面快捷方式
    SectionSetFlags ${SecStartMenu} 1
    SectionSetFlags ${SecDesktop} 1
FunctionEnd

; ---------- 卸载区段 ----------
Section "Uninstall"
    SetShellVarContext all
    Delete "$SMPROGRAMS\MaaNTE\MaaNTE.lnk"
    Delete "$SMPROGRAMS\MaaNTE\Uninstall.lnk"
    RMDir  "$SMPROGRAMS\MaaNTE"
    Delete "$DESKTOP\MaaNTE.lnk"
    RMDir /r "$INSTDIR"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaaNTE"
    DeleteRegKey HKLM "Software\MaaNTE"
SectionEnd
