# pack71.ps1 - print the exact harakiri_sweep command for one 2^50-key "pack" of puzzle-71's top region.
#
# Region swept: [2^71 - 2^60, 2^71)   (the top 2^60 keys of puzzle 71)
# Split into 1024 packs of 2^50 keys each. Pack 0 = lowest part of that region,
# pack 1023 = highest part (keys closest to 2^71). If you think the key is "at the end",
# start at pack 1023 and work DOWN (1023, 1022, 1021, ...).
#
# Usage:
#   .\pack71.ps1            # pack 0
#   .\pack71.ps1 1023       # the last pack (closest to 2^71)
#   .\pack71.ps1 512        # any index 0..1023

param([int]$Pack = 0)

Add-Type -AssemblyName System.Numerics

# .NET's BigInteger.ToString("X") occasionally emits a spurious leading zero (e.g. for exactly 2^71).
# Trim it so displayed hex is the canonical form you'd type by hand.
function Hex($v) { $h = $v.ToString("X"); if ($h.Length -gt 1) { $h = $h.TrimStart([char]'0') }; if ($h -eq "") { "0" } else { $h } }

$base = [System.Numerics.BigInteger]::Pow(2,71) - [System.Numerics.BigInteger]::Pow(2,60)  # region start
$step = [System.Numerics.BigInteger]::Pow(2,50)                                            # one pack size (keys)

if ($Pack -lt 0 -or $Pack -gt 1023) { Write-Error "pack index must be 0..1023"; exit 1 }

$start = $base + $step * $Pack
$end   = $start + $step
$count = [System.Numerics.BigInteger]::Pow(2,50).ToString()                                # 1125899906842624

Write-Output ("pack {0} / 1023" -f $Pack)
Write-Output ("  start hex : " + (Hex $start))
Write-Output ("  end   hex : " + (Hex $end) + "   (exclusive)")
Write-Output ("  count dec : " + $count)
Write-Output ""
Write-Output (".\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range {0}:+{1} --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 200 --sections 3 --gpui 0,1,2" -f (Hex $start), $count)
