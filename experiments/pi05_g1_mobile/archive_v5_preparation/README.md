# V5 candidate — not adopted, no actual archive/preflight run

Source SHA256 `bf382d0bbfccc29547a248777a89aa64086f0f5404060f87ab65847e4e56d098`. Eight tiny CPU tests passed. V4 remains untouched, exact diff is `v4_to_v5_exact.diff`.

V4 readiness/full60/independentaudit/ownedclosure/coverage/requiredSHA andpermanentimmutable sameUIDpayloadhardlink rules are preserved. Only whitelisttext>=65536B isgzip6/mtime0; symlinktext isv4plaincopy. MP4/PNG/models/NPZ nevergzip.

Storedtext uses flat `text_gzip/<SHA256(logical relativepath)>.gz` to prevent both foo.json.gz suffixcollision and nested-file/directorycollision. Inventory explicitlyretains logical `retained/...`, originalsize/SHA, storedgzpath/size/SHA androundtripSHA. Closureverification alwaysusesoriginalSHA. Gzipfilesarestaginguntilverified/inventoryrecorded; allfailed/partialfilesarekept.

Everygzipheader/body/footer/flush goes through24GiBfloor+remainingmandatoryplaincopy checks. No compressionratio isassumed. Typedreceiptcontrolspace isreservedinaddition; plaincopy chunks andmetadatawritesareguarded too. Source strictctime/nlink checks applyonlycompressedtext, preservingv4self-hardlink behavior.

## Final adoption requires another Root decision

No decision filewascreated. Finalarchive rejectsabsence of plan.text_gzip_decision={path,sha256}. ItsJSONmusthave schema `pi05_g1_mobile_archive_text_gzip_v5_decision_v1`, root_reviewed=true, adopted=true, tool_sha256=`bf382d0bbfccc29547a248777a89aa64086f0f5404060f87ab65847e4e56d098`, reserve_bytes=25769803776. Thispreparation receiptisnotapproval.

Archive CLI (documentation only):

```bash
python -m experiments.pi05_g1_mobile.archive_mobile_v5 --plan ROOT_NEW_ADOPTED_PLAN.json --output /home/ethan/ProjectBackups/2026-10-17/NEW_ARCHIVE --phase final
```

Restore CLI (documentation only):

```bash
python -m experiments.pi05_g1_mobile.archive_mobile_v5 --restore-from /ABS/V5_ARCHIVE --output /ABS/ENTIRELY_NEW_RESTORE
```

Restore verifies V5receipt/inventorySHA, safeoriginal/storedpaths, allstoredgzSHA, boundedstreameduncompression andoriginalsize/SHA. Plainmedia/modelbytescopiedindependently; no sourcecodeexecuted or modeldeserialized. Outputfreshrootoutsidearchive; failureswrite typedpartial andretainfiles. Diskcapacityuncompressed+24GiBmustfit.

Externaldiskconsumption/races andSIGKILL/hardwarefailure canpreventa finalJSON despitecontrolreserve; errorsarestderr-reportedwithoutmaskingoriginalerror. No guaranteedcapacity/eviction/compressionratio claim.
