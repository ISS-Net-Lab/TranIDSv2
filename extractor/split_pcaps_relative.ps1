# ============================
# CONFIG
# ============================

# Resolve data and local tools from the project root.
$ROOT = Split-Path -Parent $PSScriptRoot

$NORMAL_DIR = Join-Path $ROOT "normal_pcaps"
$ATTACK_DIR = Join-Path $ROOT "normal_attack_pcaps"

# SplitCap.exe is expected in the project root.
$SPLITCAP = Join-Path $ROOT "SplitCap.exe"

$FLOW_DIR_NAME = "flows"

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
Write-Host "TON_IoT PCAP Flow Splitting"
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

Write-Host "SplitCap:"
Write-Host "  $SPLITCAP"

Write-Host ""


# ============================
# CHECK SPLITCAP
# ============================

if (!(Test-Path $SPLITCAP)) {

    Write-Host "[ERROR] SplitCap.exe not found:"
    Write-Host "  $SPLITCAP"

    exit 1
}

Write-Host "[FOUND] SplitCap.exe"
Write-Host "  $SPLITCAP"
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


# Stop only when BOTH folders are missing.

if (!$normalExists -and !$attackExists) {

    Write-Host "[ERROR] Neither normal nor attack PCAP folder exists."

    exit 1
}


# ============================
# COLLECT ORIGINAL PCAPS
# ============================

# Expected layout:
#
# <current folder>\
#
# +-- normal_pcaps\
# |   +-- xxx.pcap
# |   +-- ...
# |
# +-- normal_attack_pcaps\
#     +-- normal_DDoS\
#     |   +-- xxx.pcap
#     |   +-- ...
#     |
#     +-- normal_DoS\
#     +-- normal_scanning\
#     +-- ...
#
# We only inspect original-data levels.
# Generated files inside "flows" will NOT be split again.

$captures = New-Object System.Collections.Generic.List[System.IO.FileInfo]


# ============================
# NORMAL PCAPS
# ============================

Write-Host "============================================================"
Write-Host "[1] Searching normal PCAP files"
Write-Host "============================================================"

$normalCaptures = @()


if ($normalExists) {

    $normalCaptures = @(
        Get-ChildItem -Path $NORMAL_DIR -File |
            Where-Object {
                $EXTS -contains $_.Extension.ToLower()
            }
    )


    if ($normalCaptures.Count -eq 0) {

        Write-Host "[WARNING] No normal PCAP files found."
    }
    else {

        Write-Host "[FOUND] $($normalCaptures.Count) normal PCAP file(s):"
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


# ============================
# ATTACK CATEGORY FOLDERS
# ============================

Write-Host "============================================================"
Write-Host "[2] Searching attack category folders"
Write-Host "============================================================"

$attackCategories = @()


if ($attackExists) {

    $attackCategories = @(
        Get-ChildItem -Path $ATTACK_DIR -Directory |
            Where-Object {
                $_.Name -ne $FLOW_DIR_NAME
            }
    )


    if ($attackCategories.Count -eq 0) {

        Write-Host "[WARNING] No attack category folders found."
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


# ============================
# ATTACK PCAPS
# ============================

Write-Host "============================================================"
Write-Host "[3] Searching attack PCAP files"
Write-Host "============================================================"


foreach ($category in $attackCategories) {

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "Category:"
    Write-Host "  $($category.Name)"
    Write-Host "Folder:"
    Write-Host "  $($category.FullName)"
    Write-Host "------------------------------------------------------------"


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

Write-Host ""


# ============================
# COLLECTION SUMMARY
# ============================

Write-Host "============================================================"
Write-Host "COLLECTION SUMMARY"
Write-Host "============================================================"

Write-Host "Normal PCAPs        : $($normalCaptures.Count)"
Write-Host "Attack categories   : $($attackCategories.Count)"
Write-Host "Total PCAP files    : $($captures.Count)"

Write-Host ""


if ($captures.Count -eq 0) {

    Write-Host "[WARNING] Nothing to split."

    exit 0
}


# ============================
# LIST ALL INPUT PCAPS
# ============================

Write-Host "============================================================"
Write-Host "ALL PCAP FILES TO BE SPLIT"
Write-Host "============================================================"

$inputIndex = 0


foreach ($f in $captures) {

    $inputIndex++

    Write-Host ""
    Write-Host "[$inputIndex/$($captures.Count)]"

    Write-Host "  Name:"
    Write-Host "    $($f.Name)"

    Write-Host "  Path:"
    Write-Host "    $($f.FullName)"
}

Write-Host ""


# ============================
# RESULT TRACKING
# ============================

# Store successful PCAP information.

$successfulPcaps = New-Object System.Collections.Generic.List[object]

# Store failed PCAP information.

$failedPcaps = New-Object System.Collections.Generic.List[object]


$successCount = 0
$failedCount = 0

$totalGeneratedFlows = 0


# ============================
# SPLIT
# ============================

Write-Host "============================================================"
Write-Host "START SPLITTING"
Write-Host "============================================================"

$current = 0


foreach ($f in $captures) {

    $current++


    $pcapPath = $f.FullName
    $pcapDir = $f.DirectoryName
    $pcapName = $f.Name


    $base = [System.IO.Path]::GetFileNameWithoutExtension(
        $pcapName
    )


    # Final flow PCAPs stay directly under one flows folder:
    #
    # normal_pcaps\flows\*.pcap
    #
    # normal_attack_pcaps\<category>\flows\*.pcap
    #
    # SplitCap first writes to a temporary folder.
    # We prefix every generated file with the source PCAP basename.

    $flowsRoot = Join-Path $pcapDir $FLOW_DIR_NAME

    $tempOut = Join-Path $flowsRoot (
        "__tmp_" + $base
    )


    # ============================
    # DISPLAY CURRENT FILE
    # ============================

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$current/$($captures.Count)] SPLITTING"
    Write-Host "============================================================"

    Write-Host "PCAP name:"
    Write-Host "  $pcapName"

    Write-Host ""

    Write-Host "Input path:"
    Write-Host "  $pcapPath"

    Write-Host ""

    Write-Host "Final flows folder:"
    Write-Host "  $flowsRoot"

    Write-Host ""

    Write-Host "Temporary output:"
    Write-Host "  $tempOut"

    Write-Host ""


    # ============================
    # CREATE OUTPUT DIRECTORIES
    # ============================

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $flowsRoot |
        Out-Null


    if (Test-Path $tempOut) {

        Remove-Item `
            $tempOut `
            -Recurse `
            -Force
    }


    New-Item `
        -ItemType Directory `
        -Force `
        -Path $tempOut |
        Out-Null


    # ============================
    # EXECUTE SPLITCAP
    # ============================

    Write-Host "[RUN] SplitCap"
    Write-Host ""


    & $SPLITCAP `
        -r $pcapPath `
        -s flow `
        -o $tempOut


    # ============================
    # SUCCESS
    # ============================

    if ($LASTEXITCODE -eq 0) {

        # Get all generated flow files.

        $generated = @(
            Get-ChildItem `
                -Path $tempOut `
                -Recurse `
                -File
        )


        $movedCount = 0


        # ============================
        # MOVE GENERATED FLOWS
        # ============================

        foreach ($g in $generated) {

            # Prefix flow filename with original PCAP basename.

            $newName = (
                $base +
                "__" +
                $g.Name
            )


            $dest = Join-Path `
                $flowsRoot `
                $newName


            # ============================
            # DUPLICATE NAME FALLBACK
            # ============================

            if (Test-Path $dest) {

                $stem = [System.IO.Path]::GetFileNameWithoutExtension(
                    $newName
                )

                $ext = [System.IO.Path]::GetExtension(
                    $newName
                )

                $n = 1


                do {

                    $dest = Join-Path `
                        $flowsRoot `
                        (
                            "{0}_dup{1}{2}" -f `
                            $stem, `
                            $n, `
                            $ext
                        )

                    $n++

                } while (Test-Path $dest)
            }


            Move-Item `
                $g.FullName `
                $dest


            $movedCount++
        }


        # ============================
        # DELETE TEMP
        # ============================

        if (Test-Path $tempOut) {

            Remove-Item `
                $tempOut `
                -Recurse `
                -Force
        }


        # ============================
        # DELETE ORIGINAL PCAP
        # ============================

        Remove-Item `
            $pcapPath `
            -Force


        # ============================
        # SUCCESS RECORD
        # ============================

        $successCount++

        $totalGeneratedFlows += $movedCount


        $successfulPcaps.Add(
            [PSCustomObject]@{
                Name       = $pcapName
                Path       = $pcapPath
                FlowFolder = $flowsRoot
                FlowCount  = $movedCount
            }
        )


        # ============================
        # DISPLAY SUCCESS
        # ============================

        Write-Host ""
        Write-Host "[SUCCESS] PCAP successfully processed:"
        Write-Host "  Name       : $pcapName"
        Write-Host "  Source     : $pcapPath"
        Write-Host "  Flow folder: $flowsRoot"
        Write-Host "  Flow files : $movedCount"
        Write-Host "  Original PCAP deleted."
    }

    # ============================
    # FAILED
    # ============================

    else {

        $failedCount++


        $failedPcaps.Add(
            [PSCustomObject]@{
                Name = $pcapName
                Path = $pcapPath
            }
        )


        Write-Host ""
        Write-Host "[FAILED] SplitCap failed:"
        Write-Host "  Name : $pcapName"
        Write-Host "  Path : $pcapPath"

        Write-Host "  Original PCAP was kept."


        if (Test-Path $tempOut) {

            Remove-Item `
                $tempOut `
                -Recurse `
                -Force
        }
    }
}


# ============================
# SUCCESSFUL PCAP LIST
# ============================

Write-Host ""
Write-Host "============================================================"
Write-Host "SUCCESSFULLY PROCESSED PCAP FILES"
Write-Host "============================================================"


if ($successfulPcaps.Count -eq 0) {

    Write-Host "No PCAP files were successfully processed."
}
else {

    $i = 0

    foreach ($item in $successfulPcaps) {

        $i++

        Write-Host ""
        Write-Host "[$i/$($successfulPcaps.Count)]"

        Write-Host "  PCAP name:"
        Write-Host "    $($item.Name)"

        Write-Host "  Original path:"
        Write-Host "    $($item.Path)"

        Write-Host "  Generated flows:"
        Write-Host "    $($item.FlowCount)"

        Write-Host "  Flow folder:"
        Write-Host "    $($item.FlowFolder)"
    }
}


# ============================
# FAILED PCAP LIST
# ============================

if ($failedPcaps.Count -gt 0) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "FAILED PCAP FILES"
    Write-Host "============================================================"


    $i = 0


    foreach ($item in $failedPcaps) {

        $i++

        Write-Host ""
        Write-Host "[$i/$($failedPcaps.Count)]"

        Write-Host "  Name:"
        Write-Host "    $($item.Name)"

        Write-Host "  Path:"
        Write-Host "    $($item.Path)"
    }
}


# ============================
# FINAL SUMMARY
# ============================

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL DONE"
Write-Host "============================================================"

Write-Host "Input PCAP files      : $($captures.Count)"
Write-Host "Successful PCAPs      : $successCount"
Write-Host "Failed PCAPs          : $failedCount"
Write-Host "Generated flow files  : $totalGeneratedFlows"

Write-Host "============================================================"
