# ============================
# CONFIG
# ============================

# Resolve data and local tools from the project root.
$ROOT = Split-Path -Parent $PSScriptRoot

$NORMAL_DIR = Join-Path $ROOT "normal_pcaps"
$ATTACK_DIR = Join-Path $ROOT "normal_attack_pcaps"

$OUT = Join-Path $ROOT "_packed_7z"
$FLOW_DIR_NAME = "flows"


# ============================
# PACK SETTINGS
# ============================

# Approximately this many UNCOMPRESSED GB per archive.
$TARGET_PART_SIZE_GB = 20

# Optional manual overrides.
$PARTS_OVERRIDE = @{
    # "normal_pcaps" = 5
    # "Injection_normal" = 2
    # "normal_DDoS" = 4
}

# 7-Zip compression settings.
$MX = 5
$SOLID = "on"

# If true, final summary also lists PCAP files contained
# in each successfully generated archive.
$LIST_SUCCESSFUL_PCAPS = $true

# IMPORTANT:
# If true, after ALL 7z parts for one source folder are created
# AND each archive passes "7z t" integrity verification,
# delete that whole source folder before continuing to the next folder.
#
# Examples:
#   normal_pcaps\
#   normal_attack_pcaps\normal_DDoS\
#   normal_attack_pcaps\normal_DoS\
#
# The parent folder "normal_attack_pcaps" itself is NOT deleted.
$DELETE_SOURCE_GROUP_AFTER_SUCCESS = $true


# ============================
# LOCATE 7-ZIP
# ============================

$SEVENZIP = $null

# 1. 7z.exe in the project root.
$local7z = Join-Path $ROOT "7z.exe"

if (Test-Path $local7z) {

    $SEVENZIP = $local7z
}
else {

    # 2. 7z.exe in PATH.
    $cmd = Get-Command 7z.exe -ErrorAction SilentlyContinue

    if ($cmd) {

        $SEVENZIP = $cmd.Source
    }
    else {

        # 3. Standard 7-Zip installation.
        $standard7z = Join-Path $env:ProgramFiles "7-Zip\7z.exe"

        if (Test-Path $standard7z) {

            $SEVENZIP = $standard7z
        }
    }
}


# ============================
# RESULT TRACKING
# ============================

$script:SuccessfulArchives = New-Object System.Collections.ArrayList
$script:FailedArchives = New-Object System.Collections.ArrayList
$script:ProcessedGroups = New-Object System.Collections.ArrayList
$script:DeletedSourceGroups = New-Object System.Collections.ArrayList
$script:KeptSourceGroups = New-Object System.Collections.ArrayList

$script:TotalArchiveAttempts = 0
$script:TotalPackedFiles = 0
$script:TotalPackedBytes = [Int64]0


# ============================
# BASIC INFORMATION
# ============================

Write-Host ""
Write-Host "============================================================"
Write-Host "TON_IoT Flow PCAP 7z Packing"
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

Write-Host "Output root:"
Write-Host "  $OUT"
Write-Host ""

Write-Host "Target uncompressed part size:"
Write-Host "  $TARGET_PART_SIZE_GB GB"
Write-Host ""

Write-Host "Delete source folder after complete + verified packing:"
Write-Host "  $DELETE_SOURCE_GROUP_AFTER_SUCCESS"
Write-Host ""


# ============================
# CHECK 7-ZIP
# ============================

if (!$SEVENZIP -or !(Test-Path $SEVENZIP)) {

    Write-Host "[ERROR] 7z.exe was not found."
    Write-Host ""
    Write-Host "Install 7-Zip, add 7z.exe to PATH,"
    Write-Host "or place 7z.exe in the project root."

    exit 1
}

Write-Host "[FOUND] 7-Zip:"
Write-Host "  $SEVENZIP"
Write-Host ""


# ============================
# CHECK INPUT FOLDERS
# ============================

$normalExists = Test-Path $NORMAL_DIR
$attackExists = Test-Path $ATTACK_DIR


if ($normalExists) {

    Write-Host "[FOUND] Normal PCAP folder:"
    Write-Host "  $NORMAL_DIR"
}
else {

    Write-Host "[WARNING] Normal PCAP folder not found:"
    Write-Host "  $NORMAL_DIR"
    Write-Host "  -> Normal packing will be skipped."
}

Write-Host ""


if ($attackExists) {

    Write-Host "[FOUND] Attack PCAP root folder:"
    Write-Host "  $ATTACK_DIR"
}
else {

    Write-Host "[WARNING] Attack PCAP root folder not found:"
    Write-Host "  $ATTACK_DIR"
    Write-Host "  -> Attack packing will be skipped."
}

Write-Host ""


# Only stop when BOTH folders are absent.
if (!$normalExists -and !$attackExists) {

    Write-Host "[ERROR] Neither normal nor attack PCAP folder exists."

    exit 1
}


# Create output root.
New-Item -ItemType Directory -Force -Path $OUT | Out-Null


# ============================================================
# HELPER: HUMAN-READABLE SIZE
# ============================================================

function SizeToHuman([Int64]$bytes) {

    if ($bytes -ge 1TB) {

        return "{0:N2} TB" -f ($bytes / 1TB)
    }

    if ($bytes -ge 1GB) {

        return "{0:N2} GB" -f ($bytes / 1GB)
    }

    if ($bytes -ge 1MB) {

        return "{0:N2} MB" -f ($bytes / 1MB)
    }

    if ($bytes -ge 1KB) {

        return "{0:N2} KB" -f ($bytes / 1KB)
    }

    return "$bytes B"
}


# ============================================================
# HELPER: CALCULATE PART COUNT
# ============================================================

function Get-PartCount(
    [string]$groupName,
    [Int64]$totalBytes,
    [int]$fileCount
) {

    # Manual override.
    if ($PARTS_OVERRIDE.ContainsKey($groupName)) {

        $parts = [int]$PARTS_OVERRIDE[$groupName]
    }
    else {

        $targetBytes = [double]$TARGET_PART_SIZE_GB * [double](1GB)

        $parts = [int][Math]::Ceiling(
            [double]$totalBytes / $targetBytes
        )
    }


    if ($parts -lt 1) {

        $parts = 1
    }


    # Avoid empty archives.
    if ($parts -gt $fileCount) {

        $parts = $fileCount
    }


    return $parts
}


# ============================================================
# HELPER: BALANCED FILE SPLITTING
# ============================================================
#
# Largest-first greedy bin packing.
#
# IMPORTANT:
# Uses ArrayList instead of PowerShell arrays.
# This avoids the "single object is automatically unwrapped"
# problem that caused:
#
#   $groups[$minIdx] == $null
#
# when parts = 1.
#
# ============================================================

function Split-FilesBySizeBalanced(
    $fileInfos,
    [int]$k
) {

    if ($k -lt 1) {

        throw "Split-FilesBySizeBalanced: k must be >= 1. Current value: $k"
    }


    # --------------------------------------------------------
    # Create groups
    # --------------------------------------------------------

    $groups = New-Object System.Collections.ArrayList


    for ($i = 0; $i -lt $k; $i++) {

        $fileList = New-Object System.Collections.ArrayList

        $group = [PSCustomObject]@{
            Files = $fileList
            Total = [Int64]0
        }

        [void]$groups.Add($group)
    }


    # Sanity check.
    if ($groups.Count -ne $k) {

        throw "Internal error: requested $k groups but created $($groups.Count)."
    }


    # --------------------------------------------------------
    # Sort largest files first
    # --------------------------------------------------------

    $sorted = @(
        $fileInfos | Sort-Object Length -Descending
    )


    # --------------------------------------------------------
    # Assign every file to currently smallest group
    # --------------------------------------------------------

    foreach ($f in $sorted) {

        $minIdx = 0


        for ($i = 1; $i -lt $groups.Count; $i++) {

            if ($groups[$i].Total -lt $groups[$minIdx].Total) {

                $minIdx = $i
            }
        }


        # Safety checks.
        if ($null -eq $groups[$minIdx]) {

            throw "Internal error: groups[$minIdx] is null."
        }


        if ($null -eq $groups[$minIdx].Files) {

            throw "Internal error: groups[$minIdx].Files is null."
        }


        [void]$groups[$minIdx].Files.Add($f)

        $groups[$minIdx].Total = [Int64](
            [Int64]$groups[$minIdx].Total +
            [Int64]$f.Length
        )
    }


    # IMPORTANT:
    # Prevent PowerShell from enumerating/unwrapping the ArrayList.
    Write-Output -NoEnumerate $groups
}


# ============================================================
# PACK ONE FLOW GROUP
# ============================================================

function Pack-FlowGroup(
    [string]$groupName,
    [string]$flowDir,
    [string]$outputDir,
    [string]$sourceGroupDir
) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "CHECKING GROUP"
    Write-Host "============================================================"

    Write-Host "Group:"
    Write-Host "  $groupName"
    Write-Host ""

    Write-Host "Source group folder:"
    Write-Host "  $sourceGroupDir"
    Write-Host ""

    Write-Host "Expected flows folder:"
    Write-Host "  $flowDir"
    Write-Host ""


    # ============================
    # CHECK SOURCE DIRECTORY
    # ============================

    if (!(Test-Path $sourceGroupDir)) {

        Write-Host "[SKIP] Source group folder not found:"
        Write-Host "  $sourceGroupDir"

        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceGroupDir
                Reason    = "Source folder not found."
            }
        )

        return
    }


    # ============================
    # CHECK FLOW DIRECTORY
    # ============================

    if (!(Test-Path $flowDir)) {

        Write-Host "[SKIP] flows folder not found:"
        Write-Host "  $flowDir"

        Write-Host "[KEEP] Source folder will NOT be deleted."

        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceGroupDir
                Reason    = "flows folder not found."
            }
        )

        return
    }


    Write-Host "[FOUND] flows folder:"
    Write-Host "  $flowDir"
    Write-Host ""


    # ============================
    # FIND PCAP FILES
    # ============================

    $pcaps = @(
        Get-ChildItem `
            -Path $flowDir `
            -Recurse `
            -Filter *.pcap `
            -File
    )


    if ($pcaps.Count -eq 0) {

        Write-Host "[SKIP] No .pcap files found under:"
        Write-Host "  $flowDir"

        Write-Host "[KEEP] Source folder will NOT be deleted."

        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceGroupDir
                Reason    = "No .pcap files found in flows folder."
            }
        )

        return
    }


    # ============================
    # SIZE INFORMATION
    # ============================

    $measurement = $pcaps | Measure-Object Length -Sum


    if ($null -eq $measurement.Sum) {

        $totalBytes = [Int64]0
    }
    else {

        $totalBytes = [Int64]$measurement.Sum
    }


    $parts = Get-PartCount `
        -groupName $groupName `
        -totalBytes $totalBytes `
        -fileCount $pcaps.Count


    Write-Host "[FOUND] Flow PCAP files:"
    Write-Host "  $($pcaps.Count)"
    Write-Host ""

    Write-Host "Total uncompressed size:"
    Write-Host "  $(SizeToHuman $totalBytes)"
    Write-Host ""

    Write-Host "Archive parts:"
    Write-Host "  $parts"
    Write-Host ""


    # ============================
    # SAVE GROUP INFORMATION
    # ============================

    $groupRecord = [PSCustomObject]@{
        GroupName      = $groupName
        SourceGroupDir = $sourceGroupDir
        FlowDir        = $flowDir
        FileCount      = $pcaps.Count
        TotalBytes     = $totalBytes
        Parts          = $parts
    }

    [void]$script:ProcessedGroups.Add($groupRecord)


    # ============================
    # BALANCE FILES
    # ============================

    $groups = Split-FilesBySizeBalanced `
        -fileInfos $pcaps `
        -k $parts


    # ============================
    # DEBUG GROUPS
    # ============================

    Write-Host "------------------------------------------------------------"
    Write-Host "BALANCED GROUP RESULT"
    Write-Host "------------------------------------------------------------"

    Write-Host "Requested parts : $parts"
    Write-Host "Created groups  : $($groups.Count)"
    Write-Host ""


    for ($debugIndex = 0; $debugIndex -lt $groups.Count; $debugIndex++) {

        $debugGroup = $groups[$debugIndex]

        Write-Host (
            "  Group {0}: Files={1}, Size={2}" -f `
            ($debugIndex + 1), `
            $debugGroup.Files.Count, `
            (SizeToHuman ([Int64]$debugGroup.Total))
        )
    }

    Write-Host ""


    # ============================
    # CREATE OUTPUT DIRECTORY
    # ============================

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $outputDir |
        Out-Null


    # ============================
    # GROUP-LEVEL STATUS
    # ============================

    $groupArchiveSuccesses = 0
    $groupArchiveFailures = 0


    # ============================
    # CREATE ARCHIVES
    # ============================

    for ($i = 0; $i -lt $groups.Count; $i++) {

        $idx = $i + 1

        $archiveBase = "$groupName-$idx"

        $archivePath = Join-Path `
            $outputDir `
            ($archiveBase + ".7z")

        $listPath = Join-Path `
            $outputDir `
            ($archiveBase + "_list.txt")


        $bucket = $groups[$i]

        $bucketBytes = [Int64]$bucket.Total


        # ============================
        # BUILD RELATIVE FILE LIST
        # ============================

        $relativePaths = New-Object System.Collections.ArrayList


        foreach ($f in $bucket.Files) {

            $relativePath = $f.FullName.Substring(
                $flowDir.Length
            ).TrimStart('\')

            [void]$relativePaths.Add($relativePath)
        }


        # ============================
        # WRITE LIST FILE
        # ============================

        $relativePaths |
            Set-Content `
                -Encoding UTF8 `
                $listPath


        # Remove existing archive from previous run.
        if (Test-Path $archivePath) {

            Remove-Item `
                $archivePath `
                -Force
        }


        $script:TotalArchiveAttempts++


        # ============================
        # DISPLAY PART
        # ============================

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "PACKING PART $idx / $($groups.Count)"
        Write-Host "============================================================"

        Write-Host "Group:"
        Write-Host "  $groupName"
        Write-Host ""

        Write-Host "Archive filename:"
        Write-Host "  $archiveBase.7z"
        Write-Host ""

        Write-Host "PCAP files:"
        Write-Host "  $($bucket.Files.Count)"
        Write-Host ""

        Write-Host "Uncompressed size:"
        Write-Host "  $(SizeToHuman $bucketBytes)"
        Write-Host ""

        Write-Host "Output:"
        Write-Host "  $archivePath"
        Write-Host ""


        # ============================
        # RUN 7-ZIP CREATE
        # ============================

        $createCode = -1

        Push-Location $flowDir

        try {

            & $SEVENZIP `
                a `
                -t7z `
                "-mx=$MX" `
                "-ms=$SOLID" `
                -scsUTF-8 `
                $archivePath `
                ("@" + $listPath)

            $createCode = $LASTEXITCODE
        }
        finally {

            Pop-Location
        }


        # ============================
        # VERIFY ARCHIVE
        # ============================

        $verifyCode = -1
        $archiveVerified = $false

        if (($createCode -eq 0) -and (Test-Path $archivePath)) {

            Write-Host ""
            Write-Host "[VERIFY] Testing archive integrity:"
            Write-Host "  $archivePath"

            & $SEVENZIP `
                t `
                -y `
                $archivePath

            $verifyCode = $LASTEXITCODE

            if ($verifyCode -eq 0) {

                $archiveVerified = $true

                Write-Host "[VERIFY OK] Archive integrity test passed."
            }
            else {

                Write-Host "[VERIFY FAILED] Archive integrity test failed."
            }
        }


        # ============================
        # SUCCESS
        # ============================

        if (
            ($createCode -eq 0) -and
            (Test-Path $archivePath) -and
            $archiveVerified
        ) {

            $archiveFile = Get-Item $archivePath


            $successRecord = [PSCustomObject]@{

                GroupName = $groupName

                ArchiveName = $archiveFile.Name

                ArchivePath = $archiveFile.FullName

                PCAPCount = $bucket.Files.Count

                SourceBytes = $bucketBytes

                ArchiveBytes = [Int64]$archiveFile.Length

                PackedFiles = @($relativePaths)
            }


            [void]$script:SuccessfulArchives.Add(
                $successRecord
            )


            $script:TotalPackedFiles = (
                $script:TotalPackedFiles +
                $bucket.Files.Count
            )


            $script:TotalPackedBytes = [Int64](
                [Int64]$script:TotalPackedBytes +
                [Int64]$bucketBytes
            )


            $groupArchiveSuccesses++


            Write-Host ""
            Write-Host "[SUCCESS] Archive created and verified successfully."

            Write-Host "  Filename:"
            Write-Host "    $($archiveFile.Name)"

            Write-Host "  Full path:"
            Write-Host "    $($archiveFile.FullName)"

            Write-Host "  PCAP files:"
            Write-Host "    $($bucket.Files.Count)"

            Write-Host "  Source size:"
            Write-Host "    $(SizeToHuman $bucketBytes)"

            Write-Host "  Compressed size:"
            Write-Host "    $(SizeToHuman ([Int64]$archiveFile.Length))"
        }

        # ============================
        # FAILED
        # ============================

        else {

            $groupArchiveFailures++


            if ($createCode -ne 0) {

                $failureReason = "7-Zip archive creation failed."
                $failureCode = $createCode
            }
            elseif (!(Test-Path $archivePath)) {

                $failureReason = "Archive file was not created."
                $failureCode = $createCode
            }
            else {

                $failureReason = "7-Zip integrity verification failed."
                $failureCode = $verifyCode
            }


            $failedRecord = [PSCustomObject]@{

                GroupName = $groupName

                ArchiveName = ($archiveBase + ".7z")

                ArchivePath = $archivePath

                PCAPCount = $bucket.Files.Count

                ExitCode = $failureCode

                Reason = $failureReason
            }


            [void]$script:FailedArchives.Add(
                $failedRecord
            )


            Write-Host ""
            Write-Host "[FAILED] Archive was not accepted as successful."

            Write-Host "  Archive:"
            Write-Host "    $archiveBase.7z"

            Write-Host "  Path:"
            Write-Host "    $archivePath"

            Write-Host "  Reason:"
            Write-Host "    $failureReason"

            Write-Host "  Exit code:"
            Write-Host "    $failureCode"


            # Do not keep a corrupted/unverified archive.
            if ((Test-Path $archivePath) -and !$archiveVerified) {

                try {

                    Remove-Item `
                        $archivePath `
                        -Force `
                        -ErrorAction Stop

                    Write-Host "  Partial/unverified archive deleted."
                }
                catch {

                    Write-Host "  [WARNING] Could not delete partial archive:"
                    Write-Host "    $archivePath"
                }
            }
        }
    }


    # ========================================================
    # DELETE SOURCE GROUP ONLY AFTER ALL PARTS SUCCEED
    # ========================================================

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "GROUP COMPLETION CHECK"
    Write-Host "============================================================"

    Write-Host "Group:"
    Write-Host "  $groupName"

    Write-Host "Expected archive parts:"
    Write-Host "  $($groups.Count)"

    Write-Host "Successful + verified parts:"
    Write-Host "  $groupArchiveSuccesses"

    Write-Host "Failed parts:"
    Write-Host "  $groupArchiveFailures"


    $allPartsSuccessful = (
        ($groupArchiveFailures -eq 0) -and
        ($groupArchiveSuccesses -eq $groups.Count)
    )


    if (!$allPartsSuccessful) {

        Write-Host ""
        Write-Host "[KEEP] Source folder will NOT be deleted because"
        Write-Host "       one or more archive parts failed."

        Write-Host "  $sourceGroupDir"

        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceGroupDir
                Reason    = "Not all archive parts were successfully created and verified."
            }
        )

        return
    }


    if (!$DELETE_SOURCE_GROUP_AFTER_SUCCESS) {

        Write-Host ""
        Write-Host "[KEEP] All archives succeeded, but automatic deletion is disabled."

        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceGroupDir
                Reason    = "Automatic deletion is disabled."
            }
        )

        return
    }


    # ============================
    # SAFETY CHECK BEFORE DELETE
    # ============================

    $rootFull = [System.IO.Path]::GetFullPath(
        $ROOT
    ).TrimEnd('\')

    $outFull = [System.IO.Path]::GetFullPath(
        $OUT
    ).TrimEnd('\')

    $sourceFull = [System.IO.Path]::GetFullPath(
        $sourceGroupDir
    ).TrimEnd('\')

    $rootPrefix = $rootFull + '\'
    $outPrefix = $outFull + '\'


    $safeToDelete = $true


    if ($sourceFull -eq $rootFull) {

        $safeToDelete = $false
    }


    if (!$sourceFull.StartsWith(
        $rootPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {

        $safeToDelete = $false
    }


    if (
        ($sourceFull -eq $outFull) -or
        $sourceFull.StartsWith(
            $outPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {

        $safeToDelete = $false
    }


    if (!$safeToDelete) {

        Write-Host ""
        Write-Host "[SAFETY BLOCK] Refusing to delete unexpected path:"
        Write-Host "  $sourceFull"

        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceFull
                Reason    = "Safety check blocked deletion."
            }
        )

        return
    }


    # ============================
    # DELETE WHOLE SOURCE FOLDER
    # ============================

    Write-Host ""
    Write-Host "[DELETE] All archive parts are valid."
    Write-Host "         Deleting source folder:"
    Write-Host "  $sourceFull"


    try {

        Remove-Item `
            $sourceFull `
            -Recurse `
            -Force `
            -ErrorAction Stop


        if (Test-Path $sourceFull) {

            throw "Folder still exists after Remove-Item."
        }


        Write-Host "[DELETE SUCCESS] Source folder deleted."
        Write-Host "  $sourceFull"


        [void]$script:DeletedSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceFull
            }
        )
    }
    catch {

        Write-Host "[DELETE FAILED] Source folder was kept."
        Write-Host "  $sourceFull"
        Write-Host "  $($_.Exception.Message)"


        [void]$script:KeptSourceGroups.Add(
            [PSCustomObject]@{
                GroupName = $groupName
                Path      = $sourceFull
                Reason    = (
                    "Delete failed: " +
                    $_.Exception.Message
                )
            }
        )
    }
}


# ============================================================
# MAIN
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "START PACKING"
Write-Host "============================================================"


# ============================================================
# 1. NORMAL PCAPS
# ============================================================

if ($normalExists) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[1] NORMAL PCAPS"
    Write-Host "============================================================"


    $normalFlowDir = Join-Path `
        $NORMAL_DIR `
        $FLOW_DIR_NAME


    $normalOutputDir = Join-Path `
        $OUT `
        "normal_pcaps"


    Pack-FlowGroup `
        -groupName "normal_pcaps" `
        -flowDir $normalFlowDir `
        -outputDir $normalOutputDir `
        -sourceGroupDir $NORMAL_DIR
}
else {

    Write-Host ""
    Write-Host "[SKIP] normal_pcaps does not exist."
}


# ============================================================
# 2. ATTACK CATEGORIES
# ============================================================

$attackCategories = @()


if ($attackExists) {

    $attackCategories = @(
        Get-ChildItem `
            -Path $ATTACK_DIR `
            -Directory |
            Where-Object {
                $_.Name -ne $FLOW_DIR_NAME
            } |
            Sort-Object Name
    )


    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[2] ATTACK CATEGORIES"
    Write-Host "============================================================"


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


    # ============================
    # PACK EACH ATTACK CATEGORY
    # ============================

    foreach ($category in $attackCategories) {

        $categoryFlowDir = Join-Path `
            $category.FullName `
            $FLOW_DIR_NAME


        $attackOutputRoot = Join-Path `
            $OUT `
            "normal_attack_pcaps"


        $categoryOutDir = Join-Path `
            $attackOutputRoot `
            $category.Name


        Pack-FlowGroup `
            -groupName $category.Name `
            -flowDir $categoryFlowDir `
            -outputDir $categoryOutDir `
            -sourceGroupDir $category.FullName
    }
}
else {

    Write-Host ""
    Write-Host "[SKIP] normal_attack_pcaps does not exist."
}


# ============================================================
# PROCESSED GROUPS
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "PROCESSED FLOW GROUPS"
Write-Host "============================================================"


if ($script:ProcessedGroups.Count -eq 0) {

    Write-Host "No flow groups were processed."
}
else {

    $i = 0


    foreach ($item in $script:ProcessedGroups) {

        $i++


        Write-Host ""
        Write-Host "[$i/$($script:ProcessedGroups.Count)]"

        Write-Host "  Group:"
        Write-Host "    $($item.GroupName)"

        Write-Host "  Source group folder:"
        Write-Host "    $($item.SourceGroupDir)"

        Write-Host "  Flow folder:"
        Write-Host "    $($item.FlowDir)"

        Write-Host "  PCAP files:"
        Write-Host "    $($item.FileCount)"

        Write-Host "  Total source size:"
        Write-Host "    $(SizeToHuman ([Int64]$item.TotalBytes))"

        Write-Host "  Archive parts:"
        Write-Host "    $($item.Parts)"
    }
}


# ============================================================
# SUCCESSFUL ARCHIVES
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "SUCCESSFULLY CREATED ARCHIVES"
Write-Host "============================================================"


if ($script:SuccessfulArchives.Count -eq 0) {

    Write-Host "No archives were successfully created."
}
else {

    $i = 0


    foreach ($item in $script:SuccessfulArchives) {

        $i++


        Write-Host ""
        Write-Host "------------------------------------------------------------"
        Write-Host "[$i/$($script:SuccessfulArchives.Count)] SUCCESS"
        Write-Host "------------------------------------------------------------"

        Write-Host "  Group:"
        Write-Host "    $($item.GroupName)"

        Write-Host "  Archive filename:"
        Write-Host "    $($item.ArchiveName)"

        Write-Host "  Full path:"
        Write-Host "    $($item.ArchivePath)"

        Write-Host "  PCAP count:"
        Write-Host "    $($item.PCAPCount)"

        Write-Host "  Source size:"
        Write-Host "    $(SizeToHuman ([Int64]$item.SourceBytes))"

        Write-Host "  Compressed size:"
        Write-Host "    $(SizeToHuman ([Int64]$item.ArchiveBytes))"


        # ============================
        # LIST SUCCESSFULLY PACKED PCAPS
        # ============================

        if ($LIST_SUCCESSFUL_PCAPS) {

            Write-Host ""
            Write-Host "  Successfully packed PCAP files:"


            $fileNo = 0


            foreach ($packedFile in $item.PackedFiles) {

                $fileNo++

                Write-Host (
                    "    [{0}/{1}] {2}" -f `
                    $fileNo, `
                    $item.PCAPCount, `
                    $packedFile
                )
            }
        }
    }
}


# ============================================================
# FAILED ARCHIVES
# ============================================================

if ($script:FailedArchives.Count -gt 0) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "FAILED ARCHIVES"
    Write-Host "============================================================"


    $i = 0


    foreach ($item in $script:FailedArchives) {

        $i++


        Write-Host ""
        Write-Host "[$i/$($script:FailedArchives.Count)]"

        Write-Host "  Group:"
        Write-Host "    $($item.GroupName)"

        Write-Host "  Archive:"
        Write-Host "    $($item.ArchiveName)"

        Write-Host "  Path:"
        Write-Host "    $($item.ArchivePath)"

        Write-Host "  PCAP count:"
        Write-Host "    $($item.PCAPCount)"

        Write-Host "  Exit code:"
        Write-Host "    $($item.ExitCode)"

        Write-Host "  Reason:"
        Write-Host "    $($item.Reason)"
    }
}


# ============================================================
# DELETED SOURCE GROUPS
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "DELETED SOURCE GROUPS"
Write-Host "============================================================"

if ($script:DeletedSourceGroups.Count -eq 0) {

    Write-Host "No source group folders were deleted."
}
else {

    $i = 0

    foreach ($item in $script:DeletedSourceGroups) {

        $i++

        Write-Host ""
        Write-Host "[$i/$($script:DeletedSourceGroups.Count)]"

        Write-Host "  Group:"
        Write-Host "    $($item.GroupName)"

        Write-Host "  Deleted folder:"
        Write-Host "    $($item.Path)"
    }
}


# ============================================================
# KEPT SOURCE GROUPS
# ============================================================

if ($script:KeptSourceGroups.Count -gt 0) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "KEPT SOURCE GROUPS"
    Write-Host "============================================================"

    $i = 0

    foreach ($item in $script:KeptSourceGroups) {

        $i++

        Write-Host ""
        Write-Host "[$i/$($script:KeptSourceGroups.Count)]"

        Write-Host "  Group:"
        Write-Host "    $($item.GroupName)"

        Write-Host "  Folder:"
        Write-Host "    $($item.Path)"

        Write-Host "  Reason:"
        Write-Host "    $($item.Reason)"
    }
}


# ============================================================
# FINAL SUMMARY
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL DONE"
Write-Host "============================================================"

Write-Host "Processed groups       : $($script:ProcessedGroups.Count)"
Write-Host "Archive attempts       : $($script:TotalArchiveAttempts)"
Write-Host "Successful archives    : $($script:SuccessfulArchives.Count)"
Write-Host "Failed archives        : $($script:FailedArchives.Count)"
Write-Host "Packed PCAP files      : $($script:TotalPackedFiles)"
Write-Host "Packed source size     : $(SizeToHuman ([Int64]$script:TotalPackedBytes))"
Write-Host "Deleted source groups  : $($script:DeletedSourceGroups.Count)"
Write-Host "Kept source groups     : $($script:KeptSourceGroups.Count)"

Write-Host ""
Write-Host "Output root:"
Write-Host "  $OUT"

Write-Host "============================================================"
