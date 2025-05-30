; Setup script for use with Inno Setup.

[Setup]
AppName=Open Asset Import Library - SDK
AppVerName=Open Asset Import Library - SDK (v5.1.0)
DefaultDirName={pf}\Assimp
DefaultGroupName=Assimp
UninstallDisplayIcon={app}\bin\x86\assimp.exe
OutputDir=out
AppCopyright=Assimp Development Team
SetupIconFile=..\..\tools\shared\assimp_tools_icon.ico
WizardImageFile=compiler:WizModernImage-IS.BMP
WizardSmallImageFile=compiler:WizModernSmallImage-IS.BMP
LicenseFile=License.rtf
OutputBaseFileName=assimp-sdk-5.1.0-setup
VersionInfoVersion=5.1.0.0
VersionInfoTextVersion=5.1.0
VersionInfoCompany=Assimp Development Team
;ArchitecturesInstallIn64BitMode=x64

[Types]
Name: "full";    Description: "Full installation"
Name: "compact"; Description: "Compact installation, no test models or language bindings"
Name: "custom";  Description: "Custom installation"; Flags: iscustom

[Components]
Name: "main";        Description: "Main Files (32 and 64 Bit)"; Types: full compact custom; Flags: fixed
Name: "tools";       Description: "Asset Viewer & Command Line Tools (32 and 64 Bit)"; Types: full compact
Name: "help";        Description: "Help Files"; Types: full compact
Name: "samples";     Description: "Samples"; Types: full
Name: "test";        Description: "Test Models (BSD-licensed)"; Types: full
Name: "test_nonbsd"; Description: "Test Models (other (free) licenses)"; Types: full

[Run]
Filename: "{app}\stub\vc_redist.x86.exe"; Parameters: "/qb /passive /quiet"; StatusMsg: "Installing VS2017 redistributable package (32 Bit)"; Check: not IsWin64

[Files]
Source: "readme_installer.txt"; DestDir: "{app}"; Flags: isreadme

; Installer stub
Source: "vc_redist.x86.exe"; DestDir: "{app}\stub\"; Check: not IsWin64

; Common stuff
Source: "..\..\CREDITS"; DestDir: "{app}"
Source: "..\..\LICENSE"; DestDir: "{app}"
Source: "..\..\README"; DestDir: "{app}"
Source: "WEB"; DestDir: "{app}"

Source: "..\..\scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs

; x86 binaries
Source: "..\..\bin\release\assimp-vc141-mt.dll";      DestDir: "{app}\bin\x86"
Source: "..\..\bin\release\assimp_viewer.exe";        DestDir: "{app}\bin\x86"; Components: tools
Source: "C:\Windows\SysWOW64\D3DCompiler_42.dll";     DestDir: "{app}\bin\x86"; Components: tools
Source: "C:\Windows\SysWOW64\D3DX9_42.dll";           DestDir: "{app}\bin\x86"; Components: tools
Source: "..\..\bin\release\assimp.exe";               DestDir: "{app}\bin\x86"; Components: tools


; Import libraries
Source: "..\..\lib\release\assimp-vc141-mt.lib"; DestDir: "{app}\lib\x86"

; Samples
Source: "..\..\samples\*"; DestDir: "{app}\samples"; Flags: recursesubdirs; Components: samples

; Include files
Source: "..\..\include\*"; DestDir: "{app}\include"; Flags: recursesubdirs

; CMake files
Source: "..\..\cmake-modules\*"; DestDir: "{app}\cmake-modules"; Flags: recursesubdirs

[Icons]
; Name: "{group}\Assimp Manual"; Filename: "{app}\doc\AssimpDoc.chm" ; Components: help
; Name: "{group}\Assimp Command Line Manual"; Filename: "{app}\doc\AssimpCmdDoc.chm"; Components: help
; Name: "{group}\AssimpView"; Filename: "{app}\bin\x64\assimp_view.exe"; Components: tools; Check: IsWin64
; Name: "{group}\AssimpView"; Filename: "{app}\bin\x86\assimp_view.exe"; Components: tools; Check: not IsWin64
