# ============================
# CONFIG
# ============================

# Use the folder containing this .ps1 file as the project root.
$ROOT = $PSScriptRoot

$NORMAL_DIR = Join-Path $ROOT "normal_pcaps"
$ATTACK_DIR = Join-Path $ROOT "normal_attack_pcaps"


# ============================
# LOCATE EDITCAP
# ============================

# Locate editcap automatically:
# 1) editcap.exe beside this script
# 2) editcap.exe available in PATH
# 3) standard Wireshark installation under %ProgramFiles%

$EDITCAP = $null

$localEditcap = Join-Path $ROOT "editcap.exe"

if (Test-Path $localEditcap) {

    $EDITCAP = $localEditcap
}
else {

    $cmd = Get-Command editcap.exe -ErrorAction SilentlyContinue

    if ($cmd) {

        $EDITCAP = $cmd.Source
    }
    else {

        $standardEditcap = Join-Path $env:ProgramFiles "Wireshark\editcap.exe"

        if (Test-Path $standardEditcap) {

            $EDITCAP = $standardEditcap
        }
    }
}


# ============================
# SUPPORTED INPUT EXTENSIONS
# ============================

$EXTS = @(
    ".pcap",
    ".pcapng",
    ".cap"
)


# ============================
# BASIC INFORMATION
# ============================

Write-Host ""
Write-Host "============================================================"
Write-Host "TON_IoT PCAP normalization"
Write-Host "============================================================"

Write-Host "Project root:"
Write-Host "  $ROOT"

Write-Host ""

Write-Host "Expected normal folder:"
Write-Host "  $NORMAL_DIR"

Write-Host ""

Write-Host "Expected attack folder:"
Write-Host "  $ATTACK_DIR"

Write-Host ""


# ============================
# CHECK EDITCAP
# ============================

if (!$EDITCAP -or !(Test-Path $EDITCAP)) {

    Write-Host "[ERROR] editcap.exe was not found."
    Write-Host ""
    Write-Host "Install Wireshark, add editcap.exe to PATH,"
    Write-Host "or place editcap.exe beside this script."

    exit 1
}

Write-Host "[FOUND] editcap:"
Write-Host "  $EDITCAP"
Write-Host ""


# ============================
# CHECK INPUT FOLDERS
# ============================

$normalExists = Test-Path $NORMAL_DIR
$attackExists = Test-Path $ATTACK_DIR


# ---- normal_pcaps ----

if ($normalExists) {

    Write-Host "[FOUND] Normal PCAP folder:"
    Write-Host "  $NORMAL_DIR"
}
else {

    Write-Host "[WARNING] Normal PCAP folder not found:"
    Write-Host "  $NORMAL_DIR"
    Write-Host "  -> Normal PCAP processing will be skipped."
}

Write-Host ""


# ---- normal_attack_pcaps ----

if ($attackExists) {

    Write-Host "[FOUND] Attack PCAP root folder:"
    Write-Host "  $ATTACK_DIR"
}
else {

    Write-Host "[WARNING] Attack PCAP root folder not found:"
    Write-Host "  $ATTACK_DIR"
    Write-Host "  -> Attack PCAP processing will be skipped."
}

Write-Host ""


# Only stop if BOTH folders are missing.

if (!$normalExists -and !$attackExists) {

    Write-Host "[ERROR] Neither normal nor attack PCAP folder exists."
    Write-Host ""
    Write-Host "Expected at least one of:"
    Write-Host "  $NORMAL_DIR"
    Write-Host "  $ATTACK_DIR"

    exit 1
}


# ============================
# COLLECT ORIGINAL PCAPS
# ============================

# Expected layout:
#
# <current folder>\
#
# ├── normal_pcaps\
# │   ├── xxx.pcap
# │   ├── xxx.pcap
# │   └── ...
# │
# └── normal_attack_pcaps\
#     ├── normal_DDoS\
#     │   ├── xxx.pcap
#     │   └── ...
#     │
#     ├── normal_DoS\
#     ├── normal_scanning\
#     └── ...
#
# We intentionally do NOT recurse into generated "flows" folders.

$captures = New-Object System.Collections.Generic.List[System.IO.FileInfo]


# ============================================================
# 1. NORMAL PCAPS
# ============================================================

Write-Host "============================================================"
Write-Host "[1] Searching normal PCAP files"
Write-Host "============================================================"

$normalCaptures = @()


if ($normalExists) {

    Write-Host "Folder:"
    Write-Host "  $NORMAL_DIR"
    Write-Host ""


    $normalCaptures = @(
        Get-ChildItem -Path $NORMAL_DIR -File |
            Where-Object {
                $EXTS -contains $_.Extension.ToLower()
            }
    )


    if ($normalCaptures.Count -eq 0) {

        Write-Host "[WARNING] No PCAP files found in normal folder."
    }
    else {

        Write-Host "[FOUND] $($normalCaptures.Count) normal capture file(s):"
        Write-Host ""


        foreach ($f in $normalCaptures) {

            Write-Host "  [PCAP] $($f.Name)"
            Write-Host "         $($f.FullName)"

            $captures.Add($f)
        }
    }
}
else {

    Write-Host "[SKIP] Normal folder does not exist."
}

Write-Host ""


# ============================================================
# 2. ATTACK CATEGORY FOLDERS
# ============================================================

Write-Host "============================================================"
Write-Host "[2] Searching attack category folders"
Write-Host "============================================================"

$attackCategories = @()


if ($attackExists) {

    Write-Host "Attack root:"
    Write-Host "  $ATTACK_DIR"
    Write-Host ""


    $attackCategories = @(
        Get-ChildItem -Path $ATTACK_DIR -Directory |
            Where-Object {
                $_.Name -ne "flows"
            }
    )


    if ($attackCategories.Count -eq 0) {

        Write-Host "[WARNING] No attack category folders found under:"
        Write-Host "  $ATTACK_DIR"
    }
    else {

        Write-Host "[FOUND] $($attackCategories.Count) attack category folder(s):"
        Write-Host ""


        foreach ($category in $attackCategories) {

            Write-Host "  [FOLDER] $($category.Name)"
            Write-Host "           $($category.FullName)"
        }
    }
}
else {

    Write-Host "[SKIP] Attack root folder does not exist."
}

Write-Host ""


# ============================================================
# 3. PCAPS INSIDE EACH ATTACK CATEGORY
# ============================================================

Write-Host "============================================================"
Write-Host "[3] Searching PCAP files in attack categories"
Write-Host "============================================================"


if ($attackCategories.Count -eq 0) {

    Write-Host "[SKIP] No attack category folders to process."
}
else {

    foreach ($category in $attackCategories) {

        Write-Host ""
        Write-Host "------------------------------------------------------------"
        Write-Host "Category:"
        Write-Host "  $($category.Name)"
        Write-Host ""
        Write-Host "Folder:"
        Write-Host "  $($category.FullName)"
        Write-Host "------------------------------------------------------------"


        # Only files directly under category folder.
        # No recursion, therefore generated flows folders are ignored.

        $categoryCaptures = @(
            Get-ChildItem -Path $category.FullName -File |
                Where-Object {
                    $EXTS -contains $_.Extension.ToLower()
                }
        )


        if ($categoryCaptures.Count -eq 0) {

            Write-Host "  [WARNING] No PCAP files found."

            continue
        }


        Write-Host "  [FOUND] $($categoryCaptures.Count) PCAP file(s):"
        Write-Host ""


        foreach ($f in $categoryCaptures) {

            Write-Host "    [PCAP] $($f.Name)"
            Write-Host "           $($f.FullName)"

            $captures.Add($f)
        }
    }
}

Write-Host ""


# ============================
# COLLECTION SUMMARY
# ============================

Write-Host "============================================================"
Write-Host "COLLECTION SUMMARY"
Write-Host "============================================================"

Write-Host "Normal folder exists : $normalExists"
Write-Host "Attack folder exists : $attackExists"
Write-Host ""
Write-Host "Normal PCAPs         : $($normalCaptures.Count)"
Write-Host "Attack categories    : $($attackCategories.Count)"
Write-Host "Total capture files  : $($captures.Count)"

Write-Host ""


if ($captures.Count -eq 0) {

    Write-Host "[WARNING] Nothing to convert."

    exit 0
}


# ============================
# LIST ALL CAPTURES
# ============================

Write-Host "============================================================"
Write-Host "ALL PCAP FILES TO BE PROCESSED"
Write-Host "============================================================"

$index = 0


foreach ($f in $captures) {

    $index++

    Write-Host ""
    Write-Host "[$index/$($captures.Count)]"

    Write-Host "  Name:"
    Write-Host "    $($f.Name)"

    Write-Host "  Folder:"
    Write-Host "    $($f.DirectoryName)"

    Write-Host "  Full path:"
    Write-Host "    $($f.FullName)"
}

Write-Host ""


# ============================
# CONVERT
# ============================

Write-Host "============================================================"
Write-Host "START CONVERSION"
Write-Host "============================================================"

$current = 0

$successCount = 0
$failedCount = 0
$skippedCount = 0


foreach ($f in $captures) {

    $current++


    $src = $f.FullName
    $dir = $f.DirectoryName


    $base = [System.IO.Path]::GetFileNameWithoutExtension(
        $f.Name
    )


    # Final file always has .pcap extension.

    $final = Join-Path $dir ($base + ".pcap")


    # Temporary output file.

    $tmp = Join-Path $dir ($base + ".__editcap_tmp.pcap")


    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "[$current/$($captures.Count)]"

    Write-Host "Input:"
    Write-Host "  $src"

    Write-Host "Output:"
    Write-Host "  $final"

    Write-Host "------------------------------------------------------------"


    # ============================
    # Prevent accidental overwrite
    # ============================

    # If source is:
    #
    # abc.pcapng
    #
    # but:
    #
    # abc.pcap
    #
    # already exists, do not overwrite it.

    if (($src -ne $final) -and (Test-Path $final)) {

        Write-Host "[SKIP] Target already exists:"
        Write-Host "  $final"

        $skippedCount++

        continue
    }


    # Remove stale temporary file.

    if (Test-Path $tmp) {

        Remove-Item $tmp -Force
    }


    # ============================
    # EDITCAP
    # ============================

    Write-Host "Converting to classic PCAP..."


    & $EDITCAP -F pcap $src $tmp


    # ============================
    # SUCCESS
    # ============================

    if ($LASTEXITCODE -eq 0) {

        # Remove original input file.

        Remove-Item $src -Force


        # Rename temporary PCAP to final PCAP.

        Move-Item $tmp $final


        Write-Host "[OK]"
        Write-Host "  $final"


        $successCount++
    }

    # ============================
    # FAILED
    # ============================

    else {

        Write-Host "[FAILED]"

        Write-Host "Original file kept:"
        Write-Host "  $src"


        if (Test-Path $tmp) {

            Remove-Item $tmp -Force
        }


        $failedCount++
    }
}


# ============================
# FINAL SUMMARY
# ============================

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL DONE"
Write-Host "============================================================"

Write-Host "Total files : $($captures.Count)"
Write-Host "Success     : $successCount"
Write-Host "Skipped     : $skippedCount"
Write-Host "Failed      : $failedCount"

Write-Host "============================================================"